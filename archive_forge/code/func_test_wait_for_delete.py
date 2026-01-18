from unittest import mock
from openstack.clustering.v1 import _proxy
from openstack.clustering.v1 import action
from openstack.clustering.v1 import build_info
from openstack.clustering.v1 import cluster
from openstack.clustering.v1 import cluster_attr
from openstack.clustering.v1 import cluster_policy
from openstack.clustering.v1 import event
from openstack.clustering.v1 import node
from openstack.clustering.v1 import policy
from openstack.clustering.v1 import policy_type
from openstack.clustering.v1 import profile
from openstack.clustering.v1 import profile_type
from openstack.clustering.v1 import receiver
from openstack.clustering.v1 import service
from openstack import proxy as proxy_base
from openstack.tests.unit import test_proxy_base
@mock.patch('openstack.resource.wait_for_delete')
def test_wait_for_delete(self, mock_wait):
    mock_resource = mock.Mock()
    mock_wait.return_value = mock_resource
    self.proxy.wait_for_delete(mock_resource)
    mock_wait.assert_called_once_with(self.proxy, mock_resource, 2, 120)