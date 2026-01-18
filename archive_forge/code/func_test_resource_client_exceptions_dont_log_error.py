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
def test_resource_client_exceptions_dont_log_error(self):

    class Controller(object):

        def __init__(self, exception_to_raise):
            self.exception_to_raise = exception_to_raise

        def raise_exception(self, req, body):
            raise self.exception_to_raise()
    actions = {'action': 'raise_exception', 'body': 'data'}
    env = {'wsgiorg.routing_args': [None, actions]}
    request = wsgi.Request.blank('/tests/123', environ=env)
    request.body = b'{"foo" : "value"}'
    resource = wsgi.Resource(Controller(self.exception), wsgi.JSONRequestDeserializer(), None)
    e = self.assertRaises(self.exception_catch, resource, request)
    e = e.exc if hasattr(e, 'exc') else e
    self.assertNotIn(str(e), self.LOG.output)