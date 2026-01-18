import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_removing_an_endpoint_group_project(self):
    endpoint_group_id = self._create_valid_endpoint_group(self.DEFAULT_ENDPOINT_GROUP_URL, self.DEFAULT_ENDPOINT_GROUP_BODY)
    url = self._get_project_endpoint_group_url(endpoint_group_id, self.default_domain_project_id)
    self.put(url)
    self.delete(url)
    self.get(url, expected_status=http.client.NOT_FOUND)