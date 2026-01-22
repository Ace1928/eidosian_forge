import ipaddress
import logging
import re
from contextlib import suppress
from io import BytesIO
from time import time
from urllib.parse import urldefrag, urlunparse
from twisted.internet import defer, protocol, ssl
from twisted.internet.endpoints import TCP4ClientEndpoint
from twisted.internet.error import TimeoutError
from twisted.python.failure import Failure
from twisted.web.client import (
from twisted.web.http import PotentialDataLoss, _DataLoss
from twisted.web.http_headers import Headers as TxHeaders
from twisted.web.iweb import UNKNOWN_LENGTH, IBodyProducer
from zope.interface import implementer
from scrapy import signals
from scrapy.core.downloader.contextfactory import load_context_factory_from_settings
from scrapy.core.downloader.webclient import _parse
from scrapy.exceptions import StopDownload
from scrapy.http import Headers
from scrapy.responsetypes import responsetypes
from scrapy.utils.python import to_bytes, to_unicode
class ScrapyAgent:
    _Agent = Agent
    _ProxyAgent = ScrapyProxyAgent
    _TunnelingAgent = TunnelingAgent

    def __init__(self, contextFactory=None, connectTimeout=10, bindAddress=None, pool=None, maxsize=0, warnsize=0, fail_on_dataloss=True, crawler=None):
        self._contextFactory = contextFactory
        self._connectTimeout = connectTimeout
        self._bindAddress = bindAddress
        self._pool = pool
        self._maxsize = maxsize
        self._warnsize = warnsize
        self._fail_on_dataloss = fail_on_dataloss
        self._txresponse = None
        self._crawler = crawler

    def _get_agent(self, request, timeout):
        from twisted.internet import reactor
        bindaddress = request.meta.get('bindaddress') or self._bindAddress
        proxy = request.meta.get('proxy')
        if proxy:
            proxyScheme, proxyNetloc, proxyHost, proxyPort, proxyParams = _parse(proxy)
            scheme = _parse(request.url)[0]
            proxyHost = to_unicode(proxyHost)
            if scheme == b'https':
                proxyAuth = request.headers.get(b'Proxy-Authorization', None)
                proxyConf = (proxyHost, proxyPort, proxyAuth)
                return self._TunnelingAgent(reactor=reactor, proxyConf=proxyConf, contextFactory=self._contextFactory, connectTimeout=timeout, bindAddress=bindaddress, pool=self._pool)
            proxyScheme = proxyScheme or b'http'
            proxyURI = urlunparse((proxyScheme, proxyNetloc, proxyParams, '', '', ''))
            return self._ProxyAgent(reactor=reactor, proxyURI=to_bytes(proxyURI, encoding='ascii'), connectTimeout=timeout, bindAddress=bindaddress, pool=self._pool)
        return self._Agent(reactor=reactor, contextFactory=self._contextFactory, connectTimeout=timeout, bindAddress=bindaddress, pool=self._pool)

    def download_request(self, request):
        from twisted.internet import reactor
        timeout = request.meta.get('download_timeout') or self._connectTimeout
        agent = self._get_agent(request, timeout)
        url = urldefrag(request.url)[0]
        method = to_bytes(request.method)
        headers = TxHeaders(request.headers)
        if isinstance(agent, self._TunnelingAgent):
            headers.removeHeader(b'Proxy-Authorization')
        if request.body:
            bodyproducer = _RequestBodyProducer(request.body)
        else:
            bodyproducer = None
        start_time = time()
        d = agent.request(method, to_bytes(url, encoding='ascii'), headers, bodyproducer)
        d.addCallback(self._cb_latency, request, start_time)
        d.addCallback(self._cb_bodyready, request)
        d.addCallback(self._cb_bodydone, request, url)
        self._timeout_cl = reactor.callLater(timeout, d.cancel)
        d.addBoth(self._cb_timeout, request, url, timeout)
        return d

    def _cb_timeout(self, result, request, url, timeout):
        if self._timeout_cl.active():
            self._timeout_cl.cancel()
            return result
        if self._txresponse:
            self._txresponse._transport.stopProducing()
        raise TimeoutError(f'Getting {url} took longer than {timeout} seconds.')

    def _cb_latency(self, result, request, start_time):
        request.meta['download_latency'] = time() - start_time
        return result

    @staticmethod
    def _headers_from_twisted_response(response):
        headers = Headers()
        if response.length != UNKNOWN_LENGTH:
            headers[b'Content-Length'] = str(response.length).encode()
        headers.update(response.headers.getAllRawHeaders())
        return headers

    def _cb_bodyready(self, txresponse, request):
        headers_received_result = self._crawler.signals.send_catch_log(signal=signals.headers_received, headers=self._headers_from_twisted_response(txresponse), body_length=txresponse.length, request=request, spider=self._crawler.spider)
        for handler, result in headers_received_result:
            if isinstance(result, Failure) and isinstance(result.value, StopDownload):
                logger.debug('Download stopped for %(request)s from signal handler %(handler)s', {'request': request, 'handler': handler.__qualname__})
                txresponse._transport.stopProducing()
                txresponse._transport.loseConnection()
                return {'txresponse': txresponse, 'body': b'', 'flags': ['download_stopped'], 'certificate': None, 'ip_address': None, 'failure': result if result.value.fail else None}
        if txresponse.length == 0:
            return {'txresponse': txresponse, 'body': b'', 'flags': None, 'certificate': None, 'ip_address': None}
        maxsize = request.meta.get('download_maxsize', self._maxsize)
        warnsize = request.meta.get('download_warnsize', self._warnsize)
        expected_size = txresponse.length if txresponse.length != UNKNOWN_LENGTH else -1
        fail_on_dataloss = request.meta.get('download_fail_on_dataloss', self._fail_on_dataloss)
        if maxsize and expected_size > maxsize:
            warning_msg = 'Cancelling download of %(url)s: expected response size (%(size)s) larger than download max size (%(maxsize)s).'
            warning_args = {'url': request.url, 'size': expected_size, 'maxsize': maxsize}
            logger.warning(warning_msg, warning_args)
            txresponse._transport.loseConnection()
            raise defer.CancelledError(warning_msg % warning_args)
        if warnsize and expected_size > warnsize:
            logger.warning('Expected response size (%(size)s) larger than download warn size (%(warnsize)s) in request %(request)s.', {'size': expected_size, 'warnsize': warnsize, 'request': request})

        def _cancel(_):
            txresponse._transport._producer.abortConnection()
        d = defer.Deferred(_cancel)
        txresponse.deliverBody(_ResponseReader(finished=d, txresponse=txresponse, request=request, maxsize=maxsize, warnsize=warnsize, fail_on_dataloss=fail_on_dataloss, crawler=self._crawler))
        self._txresponse = txresponse
        return d

    def _cb_bodydone(self, result, request, url):
        headers = self._headers_from_twisted_response(result['txresponse'])
        respcls = responsetypes.from_args(headers=headers, url=url, body=result['body'])
        try:
            version = result['txresponse'].version
            protocol = f'{to_unicode(version[0])}/{version[1]}.{version[2]}'
        except (AttributeError, TypeError, IndexError):
            protocol = None
        response = respcls(url=url, status=int(result['txresponse'].code), headers=headers, body=result['body'], flags=result['flags'], certificate=result['certificate'], ip_address=result['ip_address'], protocol=protocol)
        if result.get('failure'):
            result['failure'].value.response = response
            return result['failure']
        return response