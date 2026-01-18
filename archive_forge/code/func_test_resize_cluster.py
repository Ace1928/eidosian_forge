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
@mock.patch.object(proxy_base.Proxy, '_find')
def test_resize_cluster(self, mock_find):
    mock_cluster = cluster.Cluster.new(id='FAKE_CLUSTER')
    mock_find.return_value = mock_cluster
    self._verify('openstack.clustering.v1.cluster.Cluster.resize', self.proxy.resize_cluster, method_args=['FAKE_CLUSTER'], method_kwargs={'k1': 'v1', 'k2': 'v2'}, expected_args=[self.proxy], expected_kwargs={'k1': 'v1', 'k2': 'v2'})
    mock_find.assert_called_once_with(cluster.Cluster, 'FAKE_CLUSTER', ignore_missing=False)