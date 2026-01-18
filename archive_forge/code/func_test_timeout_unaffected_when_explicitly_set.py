from unittest import mock
from oslo_config import cfg
import oslo_messaging as messaging
from oslo_messaging import conffixture as messaging_conffixture
from oslo_messaging import exceptions as oslomsg_exc
import testtools
from neutron_lib import fixture
from neutron_lib import rpc
from neutron_lib.tests import _base as base
def test_timeout_unaffected_when_explicitly_set(self):
    rpc.TRANSPORT.conf.rpc_response_timeout = 5
    ctx = self.client.prepare(topic='sandwiches', timeout=77)
    with testtools.ExpectedException(messaging.MessagingTimeout):
        ctx.call(self.call_context, 'create_pb_and_j')
    self.assertEqual(5, rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['create_pb_and_j'])
    self.assertFalse(self.sleep.called)