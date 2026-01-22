import testtools
from openstack.container_infrastructure_management.v1 import cluster_template
from openstack import exceptions
from openstack.tests.unit import base
class TestClusterTemplates(base.TestCase):

    def _compare_clustertemplates(self, exp, real):
        self.assertDictEqual(cluster_template.ClusterTemplate(**exp).to_dict(computed=False), real.to_dict(computed=False))

    def get_mock_url(self, service_type='container-infrastructure-management', base_url_append=None, append=None, resource=None):
        return super(TestClusterTemplates, self).get_mock_url(service_type=service_type, resource=resource, append=append, base_url_append=base_url_append)

    def test_list_cluster_templates_without_detail(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='clustertemplates'), json=dict(clustertemplates=[cluster_template_obj]))])
        cluster_templates_list = self.cloud.list_cluster_templates()
        self._compare_clustertemplates(cluster_template_obj, cluster_templates_list[0])
        self.assert_calls()

    def test_list_cluster_templates_with_detail(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='clustertemplates'), json=dict(clustertemplates=[cluster_template_obj]))])
        cluster_templates_list = self.cloud.list_cluster_templates(detail=True)
        self._compare_clustertemplates(cluster_template_obj, cluster_templates_list[0])
        self.assert_calls()

    def test_search_cluster_templates_by_name(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='clustertemplates'), json=dict(clustertemplates=[cluster_template_obj]))])
        cluster_templates = self.cloud.search_cluster_templates(name_or_id='fake-cluster-template')
        self.assertEqual(1, len(cluster_templates))
        self.assertEqual('fake-uuid', cluster_templates[0]['uuid'])
        self.assert_calls()

    def test_search_cluster_templates_not_found(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='clustertemplates'), json=dict(clustertemplates=[cluster_template_obj]))])
        cluster_templates = self.cloud.search_cluster_templates(name_or_id='non-existent')
        self.assertEqual(0, len(cluster_templates))
        self.assert_calls()

    def test_get_cluster_template(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='clustertemplates'), json=dict(clustertemplates=[cluster_template_obj]))])
        r = self.cloud.get_cluster_template('fake-cluster-template')
        self.assertIsNotNone(r)
        self._compare_clustertemplates(cluster_template_obj, r)
        self.assert_calls()

    def test_get_cluster_template_not_found(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='clustertemplates'), json=dict(clustertemplates=[]))])
        r = self.cloud.get_cluster_template('doesNotExist')
        self.assertIsNone(r)
        self.assert_calls()

    def test_create_cluster_template(self):
        json_response = cluster_template_obj.copy()
        kwargs = dict(name=cluster_template_obj['name'], image_id=cluster_template_obj['image_id'], keypair_id=cluster_template_obj['keypair_id'], coe=cluster_template_obj['coe'])
        self.register_uris([dict(method='POST', uri=self.get_mock_url(resource='clustertemplates'), json=json_response, validate=dict(json=kwargs))])
        response = self.cloud.create_cluster_template(**kwargs)
        self._compare_clustertemplates(json_response, response)
        self.assert_calls()

    def test_create_cluster_template_exception(self):
        self.register_uris([dict(method='POST', uri=self.get_mock_url(resource='clustertemplates'), status_code=403)])
        with testtools.ExpectedException(exceptions.ForbiddenException):
            self.cloud.create_cluster_template('fake-cluster-template')
        self.assert_calls()

    def test_delete_cluster_template(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='clustertemplates'), json=dict(clustertemplates=[cluster_template_obj])), dict(method='DELETE', uri=self.get_mock_url(resource='clustertemplates/fake-uuid'))])
        self.cloud.delete_cluster_template('fake-uuid')
        self.assert_calls()

    def test_update_cluster_template(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='clustertemplates'), json=dict(clustertemplates=[cluster_template_obj])), dict(method='PATCH', uri=self.get_mock_url(resource='clustertemplates/fake-uuid'), status_code=200, validate=dict(json=[{'op': 'replace', 'path': '/name', 'value': 'new-cluster-template'}]))])
        new_name = 'new-cluster-template'
        updated = self.cloud.update_cluster_template('fake-uuid', name=new_name)
        self.assertEqual(new_name, updated.name)
        self.assert_calls()

    def test_coe_get_cluster_template(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='clustertemplates'), json=dict(clustertemplates=[cluster_template_obj]))])
        r = self.cloud.get_cluster_template('fake-cluster-template')
        self.assertIsNotNone(r)
        self._compare_clustertemplates(cluster_template_obj, r)
        self.assert_calls()