from unittest import mock
from openstack.baremetal.v1 import _proxy
from openstack.baremetal.v1 import allocation
from openstack.baremetal.v1 import chassis
from openstack.baremetal.v1 import driver
from openstack.baremetal.v1 import node
from openstack.baremetal.v1 import port
from openstack.baremetal.v1 import port_group
from openstack.baremetal.v1 import volume_connector
from openstack.baremetal.v1 import volume_target
from openstack import exceptions
from openstack.tests.unit import base
from openstack.tests.unit import test_proxy_base
def test_timeout_and_failures_not_fail(self, mock_get):

    def _fake_get(_self, node):
        result = mock.Mock()
        result.id = getattr(node, 'id', node)
        if result.id == '1':
            result._check_state_reached.return_value = True
        elif result.id == '2':
            result._check_state_reached.side_effect = exceptions.ResourceFailure('boom')
        else:
            result._check_state_reached.return_value = False
        return result
    mock_get.side_effect = _fake_get
    result = self.proxy.wait_for_nodes_provision_state(['1', '2', '3'], 'fake state', timeout=0.001, fail=False)
    self.assertEqual(['1'], [x.id for x in result.success])
    self.assertEqual(['3'], [x.id for x in result.timeout])
    self.assertEqual(['2'], [x.id for x in result.failure])