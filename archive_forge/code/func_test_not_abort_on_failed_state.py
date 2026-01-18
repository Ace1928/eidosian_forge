from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_not_abort_on_failed_state(self, mock_fetch):

    def _get_side_effect(_self, session):
        self.node.provision_state = 'deploy failed'
        self.assertIs(session, self.session)
    mock_fetch.side_effect = _get_side_effect
    self.assertRaises(exceptions.ResourceTimeout, self.node.wait_for_provision_state, self.session, 'manageable', timeout=0.001, abort_on_failed_state=False)