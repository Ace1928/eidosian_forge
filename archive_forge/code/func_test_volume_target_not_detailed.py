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
@mock.patch.object(volume_target.VolumeTarget, 'list')
def test_volume_target_not_detailed(self, mock_list):
    result = self.proxy.volume_targets(query=1)
    self.assertIs(result, mock_list.return_value)
    mock_list.assert_called_once_with(self.proxy, query=1)