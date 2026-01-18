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
def test_to_json_with_date_format_value(self):
    fixture = {'date': datetime.datetime(1901, 3, 8, 2)}
    expected = b'{"date": "1901-03-08T02:00:00.000000"}'
    actual = wsgi.JSONResponseSerializer().to_json(fixture)
    self.assertEqual(expected, actual)