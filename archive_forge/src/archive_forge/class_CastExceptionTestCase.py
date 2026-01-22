from unittest import mock
from oslo_config import cfg
import oslo_messaging as messaging
from oslo_messaging import conffixture as messaging_conffixture
from oslo_messaging import exceptions as oslomsg_exc
import testtools
from neutron_lib import fixture
from neutron_lib import rpc
from neutron_lib.tests import _base as base
class CastExceptionTestCase(base.BaseTestCase):

    def setUp(self):
        super(CastExceptionTestCase, self).setUp()
        self.messaging_conf = messaging_conffixture.ConfFixture(CONF)
        self.messaging_conf.transport_url = 'fake://'
        self.messaging_conf.response_timeout = 0
        self.useFixture(self.messaging_conf)
        self.addCleanup(rpc.cleanup)
        rpc.init(CONF)
        rpc.TRANSPORT = mock.MagicMock()
        rpc.TRANSPORT._send.side_effect = oslomsg_exc.MessageDeliveryFailure
        rpc.TRANSPORT.conf.oslo_messaging_metrics.metrics_enabled = False
        target = messaging.Target(version='1.0', topic='testing')
        self.client = rpc.get_client(target)
        self.cast_context = mock.Mock()

    def test_cast_catches_exception(self):
        self.client.cast(self.cast_context, 'method_1')