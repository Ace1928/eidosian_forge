from openstack.container_infrastructure_management.v1 import cluster
from openstack.tests.unit import base
def test_search_coe_cluster_not_found(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='clusters'), json=dict(clusters=[coe_cluster_obj]))])
    coe_clusters = self.cloud.search_coe_clusters(name_or_id='non-existent')
    self.assertEqual(0, len(coe_clusters))
    self.assert_calls()