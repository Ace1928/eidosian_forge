import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
class HTTPServerRedirecting(http_server.HttpServer):
    """An HttpServer redirecting to another server """

    def __init__(self, request_handler=RedirectRequestHandler, protocol_version=None):
        http_server.HttpServer.__init__(self, request_handler, protocol_version=protocol_version)
        self.redirections = []

    def redirect_to(self, host, port):
        """Redirect all requests to a specific host:port"""
        self.redirections = [('(.*)', 'http://{}:{}\\1'.format(host, port), 301)]

    def is_redirected(self, path):
        """Is the path redirected by this server.

        :param path: the requested relative path

        :returns: a tuple (code, target) if a matching
             redirection is found, (None, None) otherwise.
        """
        code = None
        target = None
        for rsource, rtarget, rcode in self.redirections:
            target, match = re.subn(rsource, rtarget, path, count=1)
            if match:
                code = rcode
                break
            else:
                target = None
        return (code, target)