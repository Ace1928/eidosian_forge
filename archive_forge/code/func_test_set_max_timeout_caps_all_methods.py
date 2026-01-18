from unittest import mock
from oslo_config import cfg
import oslo_messaging as messaging
from oslo_messaging import conffixture as messaging_conffixture
from oslo_messaging import exceptions as oslomsg_exc
import testtools
from neutron_lib import fixture
from neutron_lib import rpc
from neutron_lib.tests import _base as base
def test_set_max_timeout_caps_all_methods(self):
    rpc.TRANSPORT.conf.rpc_response_timeout = 300
    rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_1'] = 100
    rpc.BackingOffClient.set_max_timeout(50)
    self.assertEqual(50, rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_1'])
    self.assertEqual(50, rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_2'])