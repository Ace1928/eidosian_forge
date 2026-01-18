import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_project_id_specified_if_include_subtree_specified(self):
    """When using include_subtree, you must specify a project ID."""
    r = self.get('/role_assignments?include_subtree=True', expected_status=http.client.BAD_REQUEST)
    error_msg = 'scope.project.id must be specified if include_subtree is also specified'
    self.assertEqual(error_msg, r.result['error']['message'])
    r = self.get('/role_assignments?scope.project.id&include_subtree=True', expected_status=http.client.BAD_REQUEST)
    self.assertEqual(error_msg, r.result['error']['message'])