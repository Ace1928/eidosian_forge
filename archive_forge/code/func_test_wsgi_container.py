from tornado.httputil import (
from tornado.routing import (
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application, HTTPError, RequestHandler
from tornado.wsgi import WSGIContainer
import typing  # noqa: F401
def test_wsgi_container(self):
    response = self.fetch('/tornado/test')
    self.assertEqual(response.body, b'/tornado/test')
    response = self.fetch('/wsgi')
    self.assertEqual(response.body, b'WSGI')