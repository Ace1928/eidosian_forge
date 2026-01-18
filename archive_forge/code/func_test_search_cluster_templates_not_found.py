import testtools
from openstack.container_infrastructure_management.v1 import cluster_template
from openstack import exceptions
from openstack.tests.unit import base
def test_search_cluster_templates_not_found(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='clustertemplates'), json=dict(clustertemplates=[cluster_template_obj]))])
    cluster_templates = self.cloud.search_cluster_templates(name_or_id='non-existent')
    self.assertEqual(0, len(cluster_templates))
    self.assert_calls()