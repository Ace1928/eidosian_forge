import re
from io import BytesIO
from urllib.parse import unquote
from twisted.internet.protocol import ClientCreator, Protocol
from twisted.protocols.ftp import CommandFailed, FTPClient
from scrapy.http import Response
from scrapy.responsetypes import responsetypes
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.python import to_bytes
class FTPDownloadHandler:
    lazy = False
    CODE_MAPPING = {'550': 404, 'default': 503}

    def __init__(self, settings):
        self.default_user = settings['FTP_USER']
        self.default_password = settings['FTP_PASSWORD']
        self.passive_mode = settings['FTP_PASSIVE_MODE']

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)

    def download_request(self, request, spider):
        from twisted.internet import reactor
        parsed_url = urlparse_cached(request)
        user = request.meta.get('ftp_user', self.default_user)
        password = request.meta.get('ftp_password', self.default_password)
        passive_mode = 1 if bool(request.meta.get('ftp_passive', self.passive_mode)) else 0
        creator = ClientCreator(reactor, FTPClient, user, password, passive=passive_mode)
        dfd = creator.connectTCP(parsed_url.hostname, parsed_url.port or 21)
        return dfd.addCallback(self.gotClient, request, unquote(parsed_url.path))

    def gotClient(self, client, request, filepath):
        self.client = client
        protocol = ReceivedDataProtocol(request.meta.get('ftp_local_filename'))
        return client.retrieveFile(filepath, protocol).addCallbacks(callback=self._build_response, callbackArgs=(request, protocol), errback=self._failed, errbackArgs=(request,))

    def _build_response(self, result, request, protocol):
        self.result = result
        protocol.close()
        headers = {'local filename': protocol.filename or '', 'size': protocol.size}
        body = to_bytes(protocol.filename or protocol.body.read())
        respcls = responsetypes.from_args(url=request.url, body=body)
        return respcls(url=request.url, status=200, body=body, headers=headers)

    def _failed(self, result, request):
        message = result.getErrorMessage()
        if result.type == CommandFailed:
            m = _CODE_RE.search(message)
            if m:
                ftpcode = m.group()
                httpcode = self.CODE_MAPPING.get(ftpcode, self.CODE_MAPPING['default'])
                return Response(url=request.url, status=httpcode, body=to_bytes(message))
        raise result.type(result.value)