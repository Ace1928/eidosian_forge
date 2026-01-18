import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_update_endpoint_group(self):
    endpoint_group = fixtures.EndpointGroup(self.client)
    self.useFixture(endpoint_group)
    new_name = fixtures.RESOURCE_NAME_PREFIX + uuid.uuid4().hex
    new_filters = {'interface': 'public'}
    new_description = uuid.uuid4().hex
    endpoint_group_ret = self.client.endpoint_groups.update(endpoint_group, name=new_name, filters=new_filters, description=new_description)
    endpoint_group.ref.update({'name': new_name, 'filters': new_filters, 'description': new_description})
    self.check_endpoint_group(endpoint_group_ret, endpoint_group.ref)