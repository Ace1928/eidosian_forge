from tornado.httputil import (
from tornado.routing import (
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application, HTTPError, RequestHandler
from tornado.wsgi import WSGIContainer
import typing  # noqa: F401
class ConnectionDelegate(HTTPServerConnectionDelegate):

    def start_request(self, server_conn, request_conn):

        class MessageDelegate(HTTPMessageDelegate):

            def __init__(self, connection):
                self.connection = connection

            def finish(self):
                response_body = b'OK'
                self.connection.write_headers(ResponseStartLine('HTTP/1.1', 200, 'OK'), HTTPHeaders({'Content-Length': str(len(response_body))}))
                self.connection.write(response_body)
                self.connection.finish()
        return MessageDelegate(request_conn)