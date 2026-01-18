from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
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