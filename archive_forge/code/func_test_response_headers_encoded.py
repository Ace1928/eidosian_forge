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
def test_response_headers_encoded(self):
    for_openstack_comrades = 'За опенстек, товарищи'

    class FakeController(object):

        def index(self, shirt, pants=None):
            return (shirt, pants)

    class FakeSerializer(object):

        def index(self, response, result):
            response.headers['unicode_test'] = for_openstack_comrades
    resource = wsgi.Resource(FakeController(), None, FakeSerializer())
    actions = {'action': 'index'}
    env = {'wsgiorg.routing_args': [None, actions]}
    request = wsgi.Request.blank('/tests/123', environ=env)
    response = resource.__call__(request)
    value = response.headers['unicode_test']
    self.assertEqual(for_openstack_comrades, value)