import uuid
from keystoneauth1 import fixture
from keystoneauth1.tests.unit import utils
def test_project_scoped(self):
    project_id = uuid.uuid4().hex
    project_name = uuid.uuid4().hex
    project_domain_id = uuid.uuid4().hex
    project_domain_name = uuid.uuid4().hex
    token = fixture.V3Token(project_id=project_id, project_name=project_name, project_domain_id=project_domain_id, project_domain_name=project_domain_name)
    self.assertEqual(project_id, token.project_id)
    self.assertEqual(project_id, token['token']['project']['id'])
    self.assertEqual(project_name, token.project_name)
    self.assertEqual(project_name, token['token']['project']['name'])
    self.assertIsNone(token.get('token', {}).get('is_domain'))
    project_domain = token['token']['project']['domain']
    self.assertEqual(project_domain_id, token.project_domain_id)
    self.assertEqual(project_domain_id, project_domain['id'])
    self.assertEqual(project_domain_name, token.project_domain_name)
    self.assertEqual(project_domain_name, project_domain['name'])