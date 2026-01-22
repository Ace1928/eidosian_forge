import uuid
import webob
from oslo_messaging.notify import middleware
from oslo_messaging.tests import utils
from unittest import mock
class FakeFailingApp(object):

    def __call__(self, env, start_response):
        raise Exception('It happens!')