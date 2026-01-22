import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import test_v3
def _get_project_endpoint_group_url(self, endpoint_group_id, project_id):
    return '/OS-EP-FILTER/endpoint_groups/%(endpoint_group_id)s/projects/%(project_id)s' % {'endpoint_group_id': endpoint_group_id, 'project_id': project_id}