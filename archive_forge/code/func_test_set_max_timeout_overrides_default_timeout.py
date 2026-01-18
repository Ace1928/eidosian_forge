from unittest import mock
from oslo_config import cfg
import oslo_messaging as messaging
from oslo_messaging import conffixture as messaging_conffixture
from oslo_messaging import exceptions as oslomsg_exc
import testtools
from neutron_lib import fixture
from neutron_lib import rpc
from neutron_lib.tests import _base as base
def test_set_max_timeout_overrides_default_timeout(self):
    rpc.TRANSPORT.conf.rpc_response_timeout = 10
    self.assertEqual(rpc.TRANSPORT.conf.rpc_response_max_timeout, rpc._BackingOffContextWrapper.get_max_timeout())
    rpc._BackingOffContextWrapper.set_max_timeout(10)
    self.assertEqual(10, rpc._BackingOffContextWrapper.get_max_timeout())