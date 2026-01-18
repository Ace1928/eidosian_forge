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
def test_resize_cluster_with_obj(self):
    mock_cluster = cluster.Cluster.new(id='FAKE_CLUSTER')
    self._verify('openstack.clustering.v1.cluster.Cluster.resize', self.proxy.resize_cluster, method_args=[mock_cluster], method_kwargs={'k1': 'v1', 'k2': 'v2'}, expected_args=[self.proxy], expected_kwargs={'k1': 'v1', 'k2': 'v2'})