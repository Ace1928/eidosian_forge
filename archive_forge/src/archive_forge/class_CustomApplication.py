from tornado.httputil import (
from tornado.routing import (
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application, HTTPError, RequestHandler
from tornado.wsgi import WSGIContainer
import typing  # noqa: F401
class CustomApplication(Application):

    def reverse_url(self, name, *args):
        return router.reverse_url(name, *args)