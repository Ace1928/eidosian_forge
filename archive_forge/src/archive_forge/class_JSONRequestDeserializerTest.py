import datetime
import gettext
import http.client as http
import os
import socket
from unittest import mock
import eventlet.patcher
import fixtures
from oslo_concurrency import processutils
from oslo_serialization import jsonutils
import routes
import webob
from glance.api.v2 import router as router_v2
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
from glance import i18n
from glance.image_cache import prefetcher
from glance.tests import utils as test_utils
class JSONRequestDeserializerTest(test_utils.BaseTestCase):

    def test_has_body_no_content_length(self):
        request = wsgi.Request.blank('/')
        request.method = 'POST'
        request.body = b'asdf'
        request.headers.pop('Content-Length')
        self.assertFalse(wsgi.JSONRequestDeserializer().has_body(request))

    def test_has_body_zero_content_length(self):
        request = wsgi.Request.blank('/')
        request.method = 'POST'
        request.body = b'asdf'
        request.headers['Content-Length'] = 0
        self.assertFalse(wsgi.JSONRequestDeserializer().has_body(request))

    def test_has_body_has_content_length(self):
        request = wsgi.Request.blank('/')
        request.method = 'POST'
        request.body = b'asdf'
        self.assertIn('Content-Length', request.headers)
        self.assertTrue(wsgi.JSONRequestDeserializer().has_body(request))

    def test_no_body_no_content_length(self):
        request = wsgi.Request.blank('/')
        self.assertFalse(wsgi.JSONRequestDeserializer().has_body(request))

    def test_from_json(self):
        fixture = '{"key": "value"}'
        expected = {'key': 'value'}
        actual = wsgi.JSONRequestDeserializer().from_json(fixture)
        self.assertEqual(expected, actual)

    def test_from_json_malformed(self):
        fixture = 'kjasdklfjsklajf'
        self.assertRaises(webob.exc.HTTPBadRequest, wsgi.JSONRequestDeserializer().from_json, fixture)

    def test_default_no_body(self):
        request = wsgi.Request.blank('/')
        actual = wsgi.JSONRequestDeserializer().default(request)
        expected = {}
        self.assertEqual(expected, actual)

    def test_default_with_body(self):
        request = wsgi.Request.blank('/')
        request.method = 'POST'
        request.body = b'{"key": "value"}'
        actual = wsgi.JSONRequestDeserializer().default(request)
        expected = {'body': {'key': 'value'}}
        self.assertEqual(expected, actual)

    def test_has_body_has_transfer_encoding(self):
        self.assertTrue(self._check_transfer_encoding(transfer_encoding='chunked'))

    def test_has_body_multiple_transfer_encoding(self):
        self.assertTrue(self._check_transfer_encoding(transfer_encoding='chunked, gzip'))

    def test_has_body_invalid_transfer_encoding(self):
        self.assertFalse(self._check_transfer_encoding(transfer_encoding='invalid', content_length=0))

    def test_has_body_invalid_transfer_encoding_no_content_len_and_body(self):
        self.assertFalse(self._check_transfer_encoding(transfer_encoding='invalid', include_body=False))

    def test_has_body_invalid_transfer_encoding_no_content_len_but_body(self):
        self.assertTrue(self._check_transfer_encoding(transfer_encoding='invalid', include_body=True))

    def test_has_body_invalid_transfer_encoding_with_content_length(self):
        self.assertTrue(self._check_transfer_encoding(transfer_encoding='invalid', content_length=5))

    def test_has_body_valid_transfer_encoding_with_content_length(self):
        self.assertTrue(self._check_transfer_encoding(transfer_encoding='chunked', content_length=1))

    def test_has_body_valid_transfer_encoding_without_content_length(self):
        self.assertTrue(self._check_transfer_encoding(transfer_encoding='chunked'))

    def _check_transfer_encoding(self, transfer_encoding=None, content_length=None, include_body=True):
        request = wsgi.Request.blank('/')
        request.method = 'POST'
        if include_body:
            request.body = b'fake_body'
        request.headers['transfer-encoding'] = transfer_encoding
        if content_length is not None:
            request.headers['content-length'] = content_length
        return wsgi.JSONRequestDeserializer().has_body(request)

    def test_get_bind_addr_default_value(self):
        expected = ('0.0.0.0', '123456')
        actual = wsgi.get_bind_addr(default_port='123456')
        self.assertEqual(expected, actual)