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
def test_to_json_with_more_deep_format(self):
    fixture = {'is_public': True, 'name': [{'name1': 'test'}]}
    expected = {'is_public': True, 'name': [{'name1': 'test'}]}
    actual = wsgi.JSONResponseSerializer().to_json(fixture)
    actual = jsonutils.loads(actual)
    for k in expected:
        self.assertEqual(expected[k], actual[k])