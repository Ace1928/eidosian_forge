import testscenarios
import time
import oslo_messaging
from oslo_messaging import rpc
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
from unittest import mock
class MockEndpoint(object):

    def oslo_rpc_server_ping(self, ctxt, **kwargs):
        return 'not_pong'