import testtools
from openstack.container_infrastructure_management.v1 import cluster_template
from openstack import exceptions
from openstack.tests.unit import base
def test_search_cluster_templates_by_name(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='clustertemplates'), json=dict(clustertemplates=[cluster_template_obj]))])
    cluster_templates = self.cloud.search_cluster_templates(name_or_id='fake-cluster-template')
    self.assertEqual(1, len(cluster_templates))
    self.assertEqual('fake-uuid', cluster_templates[0]['uuid'])
    self.assert_calls()