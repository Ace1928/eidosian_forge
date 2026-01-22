from tornado.httputil import (
from tornado.routing import (
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application, HTTPError, RequestHandler
from tornado.wsgi import WSGIContainer
import typing  # noqa: F401
class CustomRouter(ReversibleRouter):

    def __init__(self):
        super().__init__()
        self.routes = {}

    def add_routes(self, routes):
        self.routes.update(routes)

    def find_handler(self, request, **kwargs):
        if request.path in self.routes:
            app, handler = self.routes[request.path]
            return app.get_handler_delegate(request, handler)

    def reverse_url(self, name, *args):
        handler_path = '/' + name
        return handler_path if handler_path in self.routes else None