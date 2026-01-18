import testtools
from openstack.container_infrastructure_management.v1 import cluster_template
from openstack import exceptions
from openstack.tests.unit import base
def test_update_cluster_template(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='clustertemplates'), json=dict(clustertemplates=[cluster_template_obj])), dict(method='PATCH', uri=self.get_mock_url(resource='clustertemplates/fake-uuid'), status_code=200, validate=dict(json=[{'op': 'replace', 'path': '/name', 'value': 'new-cluster-template'}]))])
    new_name = 'new-cluster-template'
    updated = self.cloud.update_cluster_template('fake-uuid', name=new_name)
    self.assertEqual(new_name, updated.name)
    self.assert_calls()