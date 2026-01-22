import base64
import cgi
import errno
import http.client
import os
import re
import socket
import ssl
import sys
import time
import urllib
import urllib.request
import weakref
from urllib.parse import urlencode, urljoin, urlparse
from ... import __version__ as breezy_version
from ... import config, debug, errors, osutils, trace, transport, ui, urlutils
from ...bzr.smart import medium
from ...trace import mutter, mutter_callsite
from ...transport import ConnectedTransport, NoSuchFile, UnusableRedirect
from . import default_user_agent, ssl
from .response import handle_response
class AbstractHTTPHandler(urllib.request.AbstractHTTPHandler):
    """A custom handler for HTTP(S) requests.

    We overrive urllib.request.AbstractHTTPHandler to get a better
    control of the connection, the ability to implement new
    request types and return a response able to cope with
    persistent connections.
    """
    handler_order = 400
    _default_headers = {'Pragma': 'no-cache', 'Cache-control': 'max-age=0', 'Connection': 'Keep-Alive', 'User-agent': default_user_agent(), 'Accept': '*/*'}

    def __init__(self):
        urllib.request.AbstractHTTPHandler.__init__(self, debuglevel=DEBUG)

    def http_request(self, request):
        """Common headers setting"""
        for name, value in self._default_headers.items():
            if name not in request.headers:
                request.headers[name] = value
        return request

    def retry_or_raise(self, http_class, request, first_try):
        """Retry the request (once) or raise the exception.

        urllib.request raises exception of application level kind, we
        just have to translate them.

        http.client can raise exceptions of transport level (badly
        formatted dialog, loss of connexion or socket level
        problems). In that case we should issue the request again
        (http.client will close and reopen a new connection if
        needed).
        """
        exc_type, exc_val, exc_tb = sys.exc_info()
        if exc_type == socket.gaierror:
            origin_req_host = request.origin_req_host
            raise errors.ConnectionError("Couldn't resolve host '%s'" % origin_req_host, orig_error=exc_val)
        elif isinstance(exc_val, http.client.ImproperConnectionState):
            raise exc_val.with_traceback(exc_tb)
        elif first_try:
            if self._debuglevel >= 2:
                print('Received exception: [%r]' % exc_val)
                print('  On connection: [%r]' % request.connection)
                method = request.get_method()
                url = request.get_full_url()
                print('  Will retry, {} {!r}'.format(method, url))
            request.connection.close()
            response = self.do_open(http_class, request, False)
        else:
            if self._debuglevel >= 2:
                print('Received second exception: [%r]' % exc_val)
                print('  On connection: [%r]' % request.connection)
            if exc_type in (http.client.BadStatusLine, http.client.UnknownProtocol):
                my_exception = errors.InvalidHttpResponse(request.get_full_url(), 'Bad status line received', orig_error=exc_val)
            elif isinstance(exc_val, socket.error) and len(exc_val.args) and (exc_val.args[0] in (errno.ECONNRESET, 10053, 10054)):
                raise errors.ConnectionReset('Connection lost while sending request.')
            else:
                selector = request.selector
                my_exception = errors.ConnectionError(msg='while sending {} {}:'.format(request.get_method(), selector), orig_error=exc_val)
            if self._debuglevel >= 2:
                print('On connection: [%r]' % request.connection)
                method = request.get_method()
                url = request.get_full_url()
                print('  Failed again, {} {!r}'.format(method, url))
                print('  Will raise: [%r]' % my_exception)
            raise my_exception.with_traceback(exc_tb)
        return response

    def do_open(self, http_class, request, first_try=True):
        """See urllib.request.AbstractHTTPHandler.do_open for the general idea.

        The request will be retried once if it fails.
        """
        connection = request.connection
        if connection is None:
            raise AssertionError('Cannot process a request without a connection')
        headers = {}
        headers.update(request.header_items())
        headers.update(request.unredirected_hdrs)
        headers = {name.title(): val for name, val in headers.items()}
        try:
            method = request.get_method()
            url = request.selector
            connection._send_request(method, url, request.data, headers, encode_chunked=headers.get('Transfer-Encoding') == 'chunked')
            if 'http' in debug.debug_flags:
                trace.mutter('> {} {}'.format(method, url))
                hdrs = []
                for k, v in headers.items():
                    if k in ('Authorization', 'Proxy-Authorization'):
                        v = '<masked>'
                    hdrs.append('{}: {}'.format(k, v))
                trace.mutter('> ' + '\n> '.join(hdrs) + '\n')
            if self._debuglevel >= 1:
                print('Request sent: [%r] from (%s)' % (request, request.connection.sock.getsockname()))
            response = connection.getresponse()
            convert_to_addinfourl = True
        except (ssl.SSLError, ssl.CertificateError):
            raise
        except (socket.gaierror, http.client.BadStatusLine, http.client.UnknownProtocol, OSError, http.client.HTTPException):
            response = self.retry_or_raise(http_class, request, first_try)
            convert_to_addinfourl = False
        response.msg = response.reason
        return response
        if self._debuglevel >= 2:
            print('Receives response: %r' % response)
            print('  For: {!r}({!r})'.format(request.get_method(), request.get_full_url()))
        if convert_to_addinfourl:
            req = request
            r = response
            r.recv = r.read
            fp = socket._fileobject(r, bufsize=65536)
            resp = urllib.request.addinfourl(fp, r.msg, req.get_full_url())
            resp.code = r.status
            resp.msg = r.reason
            resp.version = r.version
            if self._debuglevel >= 2:
                print('Create addinfourl: %r' % resp)
                print('  For: {!r}({!r})'.format(request.get_method(), request.get_full_url()))
            if 'http' in debug.debug_flags:
                version = 'HTTP/%d.%d'
                try:
                    version = version % (resp.version / 10, resp.version % 10)
                except:
                    version = 'HTTP/%r' % resp.version
                trace.mutter('< {} {} {}'.format(version, resp.code, resp.msg))
                hdrs = [h.rstrip('\r\n') for h in resp.info().headers]
                trace.mutter('< ' + '\n< '.join(hdrs) + '\n')
        else:
            resp = response
        return resp