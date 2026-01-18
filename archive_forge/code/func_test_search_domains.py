from openstack import exceptions
from openstack.tests.functional import base
def test_search_domains(self):
    domain_name = self.domain_prefix + '_search'
    results = self.operator_cloud.search_domains(filters=dict(name=domain_name))
    self.assertEqual(0, len(results))
    domain = self.operator_cloud.create_domain(domain_name)
    self.assertEqual(domain_name, domain['name'])
    results = self.operator_cloud.search_domains(filters=dict(name=domain_name))
    self.assertEqual(1, len(results))
    self.assertEqual(domain_name, results[0]['name'])
    results = self.operator_cloud.search_domains(name_or_id=domain_name)
    self.assertEqual(1, len(results))
    self.assertEqual(domain_name, results[0]['name'])