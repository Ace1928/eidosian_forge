from openstack.container_infrastructure_management.v1 import cluster
from openstack.tests.unit import base
def test_update_coe_cluster(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='clusters'), json=dict(clusters=[coe_cluster_obj])), dict(method='PATCH', uri=self.get_mock_url(resource='clusters', append=[coe_cluster_obj['uuid']]), status_code=200, validate=dict(json=[{'op': 'replace', 'path': '/node_count', 'value': 3}]))])
    self.cloud.update_coe_cluster(coe_cluster_obj['uuid'], node_count=3)
    self.assert_calls()