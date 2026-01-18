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
@mock.patch.object(wsgi, 'translate_exception')
def test_resource_call_error_handle_localized(self, mock_translate_exception):

    class Controller(object):

        def delete(self, req, identity):
            raise webob.exc.HTTPBadRequest(explanation='Not Found')
    actions = {'action': 'delete', 'identity': 12}
    env = {'wsgiorg.routing_args': [None, actions]}
    request = wsgi.Request.blank('/tests/123', environ=env)
    message_es = 'No Encontrado'
    resource = wsgi.Resource(Controller(), wsgi.JSONRequestDeserializer(), None)
    translated_exc = webob.exc.HTTPBadRequest(message_es)
    mock_translate_exception.return_value = translated_exc
    e = self.assertRaises(webob.exc.HTTPBadRequest, resource, request)
    self.assertEqual(message_es, str(e))