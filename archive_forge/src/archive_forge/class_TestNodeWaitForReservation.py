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
class TestNodeWaitForReservation(base.TestCase):

    def setUp(self):
        super(TestNodeWaitForReservation, self).setUp()
        self.session = mock.Mock(spec=adapter.Adapter)
        self.session.default_microversion = '1.6'
        self.session.log = mock.Mock()
        self.node = node.Node(**FAKE)

    def test_no_reservation(self, mock_fetch):
        self.node.reservation = None
        node = self.node.wait_for_reservation(None)
        self.assertIs(node, self.node)
        self.assertFalse(mock_fetch.called)

    def test_reservation(self, mock_fetch):
        self.node.reservation = 'example.com'

        def _side_effect(node, session):
            if self.node.reservation == 'example.com':
                self.node.reservation = 'example2.com'
            else:
                self.node.reservation = None
        mock_fetch.side_effect = _side_effect
        node = self.node.wait_for_reservation(self.session)
        self.assertIs(node, self.node)
        self.assertEqual(2, mock_fetch.call_count)

    def test_timeout(self, mock_fetch):
        self.node.reservation = 'example.com'
        self.assertRaises(exceptions.ResourceTimeout, self.node.wait_for_reservation, self.session, timeout=0.001)
        mock_fetch.assert_called_with(self.node, self.session)