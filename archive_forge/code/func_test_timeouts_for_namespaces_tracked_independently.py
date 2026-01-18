from unittest import mock
from oslo_config import cfg
import oslo_messaging as messaging
from oslo_messaging import conffixture as messaging_conffixture
from oslo_messaging import exceptions as oslomsg_exc
import testtools
from neutron_lib import fixture
from neutron_lib import rpc
from neutron_lib.tests import _base as base
def test_timeouts_for_namespaces_tracked_independently(self):
    rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['ns1.method'] = 1
    rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['ns2.method'] = 1
    for ns in ('ns1', 'ns2'):
        self.client.target.namespace = ns
        for i in range(4):
            with testtools.ExpectedException(messaging.MessagingTimeout):
                self.client.call(self.call_context, 'method')
    timeouts = [call[1]['timeout'] for call in rpc.TRANSPORT._send.call_args_list]
    self.assertEqual([1, 2, 4, 8, 1, 2, 4, 8], timeouts)