import testtools
from saharaclient.tests.unit import base as test_base
def test_resource_str(self):
    dict = {'name': 'test', 'description': 'Changed Description'}
    resource = test_base.TestResource(None, dict)
    rstr = str(resource)
    self.assertIn(resource.resource_name, rstr)
    self.assertIn('name', rstr)
    self.assertIn('description', rstr)
    self.assertIn('Changed Description', rstr)
    self.assertNotIn('Test Description', rstr)
    self.assertIn('extra', rstr)
    self.assertNotIn('manager', rstr)