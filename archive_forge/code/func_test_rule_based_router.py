from tornado.httputil import (
from tornado.routing import (
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application, HTTPError, RequestHandler
from tornado.wsgi import WSGIContainer
import typing  # noqa: F401
def test_rule_based_router(self):
    response = self.fetch('/first_handler')
    self.assertEqual(response.body, b'first_handler: /first_handler')
    response = self.fetch('/first_handler', headers={'Host': 'www.example.com'})
    self.assertEqual(response.body, b'second_handler: /first_handler')
    response = self.fetch('/nested_handler')
    self.assertEqual(response.body, b'nested_handler: /nested_handler')
    response = self.fetch('/nested_not_found_handler')
    self.assertEqual(response.code, 404)
    response = self.fetch('/connection_delegate')
    self.assertEqual(response.body, b'OK')
    response = self.fetch('/request_callable')
    self.assertEqual(response.body, b'OK')
    response = self.fetch('/404')
    self.assertEqual(response.code, 404)