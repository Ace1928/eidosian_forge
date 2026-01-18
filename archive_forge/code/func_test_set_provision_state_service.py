from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
def test_set_provision_state_service(self):
    service_steps = [{'interface': 'deploy', 'step': 'hold'}]
    result = self.node.set_provision_state(self.session, 'service', service_steps=service_steps)
    self.assertIs(result, self.node)
    self.session.put.assert_called_once_with('nodes/%s/states/provision' % self.node.id, json={'target': 'service', 'service_steps': service_steps}, headers=mock.ANY, microversion='1.87', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)