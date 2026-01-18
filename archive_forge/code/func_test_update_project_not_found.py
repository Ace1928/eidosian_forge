import uuid
import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_update_project_not_found(self):
    project_data = self._get_project_data()
    self.register_uris([dict(method='GET', uri=self.get_mock_url(append=[project_data.project_id]), status_code=404), dict(method='GET', uri=self.get_mock_url(qs_elements=['name=' + project_data.project_id]), status_code=200, json={'projects': []})])
    with testtools.ExpectedException(exceptions.SDKException, 'Project %s not found.' % project_data.project_id):
        self.cloud.update_project(project_data.project_id)
    self.assert_calls()