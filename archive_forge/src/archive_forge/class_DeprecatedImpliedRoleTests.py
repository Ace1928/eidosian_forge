import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import roles
from testtools import matchers
class DeprecatedImpliedRoleTests(utils.ClientTestCase):

    def setUp(self):
        super(DeprecatedImpliedRoleTests, self).setUp()
        self.key = 'role'
        self.collection_key = 'roles'
        self.model = roles.Role
        self.manager = self.client.roles

    def test_implied_create(self):
        prior_id = uuid.uuid4().hex
        prior_name = uuid.uuid4().hex
        implied_id = uuid.uuid4().hex
        implied_name = uuid.uuid4().hex
        mock_response = {'role_inference': {'implies': {'id': implied_id, 'links': {'self': 'http://host/v3/roles/%s' % implied_id}, 'name': implied_name}, 'prior_role': {'id': prior_id, 'links': {'self': 'http://host/v3/roles/%s' % prior_id}, 'name': prior_name}}}
        self.stub_url('PUT', ['roles', prior_id, 'implies', implied_id], json=mock_response, status_code=201)
        with self.deprecations.expect_deprecations_here():
            manager_result = self.manager.create_implied(prior_id, implied_id)
            self.assertIsInstance(manager_result, roles.InferenceRule)
            self.assertEqual(mock_response['role_inference']['implies'], manager_result.implies)
            self.assertEqual(mock_response['role_inference']['prior_role'], manager_result.prior_role)