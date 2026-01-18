from unittest import mock
from oslo_config import cfg
import oslo_messaging as messaging
from oslo_messaging import conffixture as messaging_conffixture
from oslo_messaging import exceptions as oslomsg_exc
import testtools
from neutron_lib import fixture
from neutron_lib import rpc
from neutron_lib.tests import _base as base
def test_timeout_unchanged_on_other_exception(self):
    rpc._BackingOffContextWrapper._METHOD_TIMEOUTS['method_1'] = 1
    rpc.TRANSPORT._send.side_effect = ValueError
    with testtools.ExpectedException(ValueError):
        self.client.call(self.call_context, 'method_1')
    rpc.TRANSPORT._send.side_effect = messaging.MessagingTimeout
    with testtools.ExpectedException(messaging.MessagingTimeout):
        self.client.call(self.call_context, 'method_1')
    timeouts = [call[1]['timeout'] for call in rpc.TRANSPORT._send.call_args_list]
    self.assertEqual([1, 1], timeouts)