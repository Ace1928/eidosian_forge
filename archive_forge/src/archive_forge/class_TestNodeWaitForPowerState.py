from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
@mock.patch('time.sleep', lambda _t: None)
@mock.patch.object(node.Node, 'fetch', autospec=True)
class TestNodeWaitForPowerState(base.TestCase):

    def setUp(self):
        super(TestNodeWaitForPowerState, self).setUp()
        self.node = node.Node(**FAKE)
        self.session = mock.Mock()

    def test_success(self, mock_fetch):
        self.node.power_state = 'power on'

        def _get_side_effect(_self, session):
            self.node.power_state = 'power off'
            self.assertIs(session, self.session)
        mock_fetch.side_effect = _get_side_effect
        node = self.node.wait_for_power_state(self.session, 'power off')
        self.assertIs(node, self.node)

    def test_timeout(self, mock_fetch):
        self.node.power_state = 'power on'
        self.assertRaises(exceptions.ResourceTimeout, self.node.wait_for_power_state, self.session, 'power off', timeout=0.001)