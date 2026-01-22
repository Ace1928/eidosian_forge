import errno
import socket
import sys
import time
import urllib.parse
from http.client import BadStatusLine, HTTPConnection, NotConnected
from cheroot.test import webtest
import cherrypy
from cherrypy._cpcompat import HTTPSConnection, ntob, tonative
from cherrypy.test import helper
class LimitedRequestQueueTests(helper.CPWebCase):
    setup_server = staticmethod(setup_upload_server)

    def test_queue_full(self):
        conns = []
        overflow_conn = None
        try:
            for i in range(15):
                conn = self.HTTP_CONN(self.HOST, self.PORT)
                conn.putrequest('POST', '/upload', skip_host=True)
                conn.putheader('Host', self.HOST)
                conn.putheader('Content-Type', 'text/plain')
                conn.putheader('Content-Length', '4')
                conn.endheaders()
                conns.append(conn)
            overflow_conn = self.HTTP_CONN(self.HOST, self.PORT)
            for res in socket.getaddrinfo(self.HOST, self.PORT, 0, socket.SOCK_STREAM):
                af, socktype, proto, canonname, sa = res
                overflow_conn.sock = socket.socket(af, socktype, proto)
                overflow_conn.sock.settimeout(5)
                overflow_conn.sock.connect(sa)
                break
            overflow_conn.putrequest('GET', '/', skip_host=True)
            overflow_conn.putheader('Host', self.HOST)
            overflow_conn.endheaders()
            response = overflow_conn.response_class(overflow_conn.sock, method='GET')
            try:
                response.begin()
            except socket.error as exc:
                if exc.args[0] in socket_reset_errors:
                    pass
                else:
                    tmpl = 'Overflow conn did not get RST. Got {exc.args!r} instead'
                    raise AssertionError(tmpl.format(**locals()))
            except BadStatusLine:
                assert sys.platform == 'darwin'
            else:
                raise AssertionError('Overflow conn did not get RST ')
        finally:
            for conn in conns:
                conn.send(b'done')
                response = conn.response_class(conn.sock, method='POST')
                response.begin()
                self.body = response.read()
                self.assertBody("thanks for 'done'")
                self.assertEqual(response.status, 200)
                conn.close()
            if overflow_conn:
                overflow_conn.close()