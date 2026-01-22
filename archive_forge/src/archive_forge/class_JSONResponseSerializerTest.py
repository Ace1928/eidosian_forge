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
class JSONResponseSerializerTest(test_utils.BaseTestCase):

    def test_to_json(self):
        fixture = {'key': 'value'}
        expected = b'{"key": "value"}'
        actual = wsgi.JSONResponseSerializer().to_json(fixture)
        self.assertEqual(expected, actual)

    def test_to_json_with_date_format_value(self):
        fixture = {'date': datetime.datetime(1901, 3, 8, 2)}
        expected = b'{"date": "1901-03-08T02:00:00.000000"}'
        actual = wsgi.JSONResponseSerializer().to_json(fixture)
        self.assertEqual(expected, actual)

    def test_to_json_with_more_deep_format(self):
        fixture = {'is_public': True, 'name': [{'name1': 'test'}]}
        expected = {'is_public': True, 'name': [{'name1': 'test'}]}
        actual = wsgi.JSONResponseSerializer().to_json(fixture)
        actual = jsonutils.loads(actual)
        for k in expected:
            self.assertEqual(expected[k], actual[k])

    def test_to_json_with_set(self):
        fixture = set(['foo'])
        expected = b'["foo"]'
        actual = wsgi.JSONResponseSerializer().to_json(fixture)
        self.assertEqual(expected, actual)

    def test_default(self):
        fixture = {'key': 'value'}
        response = webob.Response()
        wsgi.JSONResponseSerializer().default(response, fixture)
        self.assertEqual(http.OK, response.status_int)
        content_types = [h for h in response.headerlist if h[0] == 'Content-Type']
        self.assertEqual(1, len(content_types))
        self.assertEqual('application/json', response.content_type)
        self.assertEqual(b'{"key": "value"}', response.body)