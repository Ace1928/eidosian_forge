from unittest import mock
from oslo_config import cfg
import oslo_messaging as messaging
from oslo_messaging import conffixture as messaging_conffixture
from oslo_messaging import exceptions as oslomsg_exc
import testtools
from neutron_lib import fixture
from neutron_lib import rpc
from neutron_lib.tests import _base as base
def test_timeout_store_defaults(self):
    self.assertEqual(rpc.TRANSPORT.conf.rpc_response_timeout, rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_1'])
    self.assertEqual(rpc.TRANSPORT.conf.rpc_response_timeout, rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_2'])
    rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_2'] = 7000
    self.assertEqual(rpc.TRANSPORT.conf.rpc_response_timeout, rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_1'])
    self.assertEqual(rpc.TRANSPORT.conf.rpc_response_timeout, rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_3'])