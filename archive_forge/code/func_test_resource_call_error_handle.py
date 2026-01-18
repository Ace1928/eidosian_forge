from unittest import mock
import fixtures
import json
from oslo_config import cfg
import socket
import webob
from heat.api.aws import exception as aws_exception
from heat.common import exception
from heat.common import wsgi
from heat.tests import common
def test_resource_call_error_handle(self):

    class Controller(object):

        def delete(self, req, identity):
            return (req, identity)
    actions = {'action': 'delete', 'id': 12, 'body': 'data'}
    env = {'wsgiorg.routing_args': [None, actions]}
    request = wsgi.Request.blank('/tests/123', environ=env)
    request.body = b'{"foo" : "value"}'
    resource = wsgi.Resource(Controller(), wsgi.JSONRequestDeserializer(), None)
    e = self.assertRaises(exception.HTTPExceptionDisguise, resource, request)
    self.assertIsInstance(e.exc, webob.exc.HTTPBadRequest)