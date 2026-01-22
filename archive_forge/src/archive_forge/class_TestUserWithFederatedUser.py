import uuid
from keystone.common import provider_api
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity.shadow_users import test_backend
from keystone.tests.unit.identity.shadow_users import test_core
from keystone.tests.unit.ksfixtures import database
class TestUserWithFederatedUser(ShadowUsersTests):

    def setUp(self):
        super(TestUserWithFederatedUser, self).setUp()
        self.useFixture(database.Database())
        self.load_backends()

    def assertFederatedDictsEqual(self, fed_dict, fed_object):
        self.assertEqual(fed_dict['idp_id'], fed_object['idp_id'])
        self.assertEqual(fed_dict['protocol_id'], fed_object['protocols'][0]['protocol_id'])
        self.assertEqual(fed_dict['unique_id'], fed_object['protocols'][0]['unique_id'])

    def test_get_user_when_user_has_federated_object(self):
        fed_dict = unit.new_federated_user_ref(idp_id=self.idp['id'], protocol_id=self.protocol['id'])
        user = self.shadow_users_api.create_federated_user(self.domain_id, fed_dict)
        user_ref = self.identity_api.get_user(user['id'])
        self.assertIn('federated', user_ref)
        self.assertEqual(1, len(user_ref['federated']))
        self.assertFederatedDictsEqual(fed_dict, user_ref['federated'][0])

    def test_create_user_with_invalid_idp_and_protocol_fails(self):
        baduser = unit.new_user_ref(domain_id=self.domain_id)
        baduser['federated'] = [{'idp_id': 'fakeidp', 'protocols': [{'protocol_id': 'nonexistent', 'unique_id': 'unknown'}]}]
        self.assertRaises(exception.ValidationError, self.identity_api.create_user, baduser)
        baduser['federated'][0]['idp_id'] = self.idp['id']
        self.assertRaises(exception.ValidationError, self.identity_api.create_user, baduser)

    def test_create_user_with_federated_attributes(self):
        user = unit.new_user_ref(domain_id=self.domain_id)
        unique_id = uuid.uuid4().hex
        user['federated'] = [{'idp_id': self.idp['id'], 'protocols': [{'protocol_id': self.protocol['id'], 'unique_id': unique_id}]}]
        self.assertRaises(exception.UserNotFound, self.shadow_users_api.get_federated_user, self.idp['id'], self.protocol['id'], unique_id)
        ref = self.identity_api.create_user(user)
        self.assertEqual(user['name'], ref['name'])
        self.assertEqual(user['federated'], ref['federated'])
        fed_user = self.shadow_users_api.get_federated_user(self.idp['id'], self.protocol['id'], unique_id)
        self.assertIsNotNone(fed_user)

    def test_update_user_with_invalid_idp_and_protocol_fails(self):
        baduser = unit.new_user_ref(domain_id=self.domain_id)
        baduser['federated'] = [{'idp_id': 'fakeidp', 'protocols': [{'protocol_id': 'nonexistent', 'unique_id': 'unknown'}]}]
        self.assertRaises(exception.ValidationError, self.identity_api.create_user, baduser)
        baduser['federated'][0]['idp_id'] = self.idp['id']
        self.assertRaises(exception.ValidationError, self.identity_api.create_user, baduser)

    def test_update_user_with_federated_attributes(self):
        user = self.shadow_users_api.create_federated_user(self.domain_id, self.federated_user)
        user = self.identity_api.get_user(user['id'])
        user = self.identity_api.update_user(user['id'], user)
        self.assertFederatedDictsEqual(self.federated_user, user['federated'][0])
        new_fed = [{'idp_id': self.idp['id'], 'protocols': [{'protocol_id': self.protocol['id'], 'unique_id': uuid.uuid4().hex}]}]
        user['federated'] = new_fed
        user = self.identity_api.update_user(user['id'], user)
        self.assertTrue('federated' in user)
        self.assertEqual(len(user['federated']), 1)
        self.assertEqual(user['federated'][0], new_fed[0])