import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
class IdentityTests(object):

    def _get_domain_fixture(self):
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        return domain

    def _set_domain_scope(self, domain_id):
        if CONF.identity.domain_specific_drivers_enabled:
            return domain_id

    def test_authenticate_bad_user(self):
        with self.make_request():
            self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, user_id=uuid.uuid4().hex, password=self.user_foo['password'])

    def test_authenticate_bad_password(self):
        with self.make_request():
            self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, user_id=self.user_foo['id'], password=uuid.uuid4().hex)

    def test_authenticate(self):
        with self.make_request():
            user_ref = PROVIDERS.identity_api.authenticate(user_id=self.user_sna['id'], password=self.user_sna['password'])
        self.user_sna.pop('password')
        self.user_sna['enabled'] = True
        self.assertUserDictEqual(self.user_sna, user_ref)

    def test_authenticate_and_get_roles_no_metadata(self):
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        del user['id']
        new_user = PROVIDERS.identity_api.create_user(user)
        role_member = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_member['id'], role_member)
        PROVIDERS.assignment_api.add_role_to_user_and_project(new_user['id'], self.project_baz['id'], role_member['id'])
        with self.make_request():
            user_ref = PROVIDERS.identity_api.authenticate(user_id=new_user['id'], password=user['password'])
        self.assertNotIn('password', user_ref)
        user.pop('password')
        self.assertLessEqual(user.items(), user_ref.items())
        role_list = PROVIDERS.assignment_api.get_roles_for_user_and_project(new_user['id'], self.project_baz['id'])
        self.assertEqual(1, len(role_list))
        self.assertIn(role_member['id'], role_list)

    def test_authenticate_if_no_password_set(self):
        id_ = uuid.uuid4().hex
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        PROVIDERS.identity_api.create_user(user)
        with self.make_request():
            self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, user_id=id_, password='password')

    def test_create_unicode_user_name(self):
        unicode_name = u'name 名字'
        user = unit.new_user_ref(name=unicode_name, domain_id=CONF.identity.default_domain_id)
        ref = PROVIDERS.identity_api.create_user(user)
        self.assertEqual(unicode_name, ref['name'])

    def test_get_user(self):
        user_ref = PROVIDERS.identity_api.get_user(self.user_foo['id'])
        self.user_foo.pop('password')
        self.assertIn('options', user_ref)
        self.assertDictEqual(self.user_foo, user_ref)

    def test_get_user_returns_required_attributes(self):
        user_ref = PROVIDERS.identity_api.get_user(self.user_foo['id'])
        self.assertIn('id', user_ref)
        self.assertIn('name', user_ref)
        self.assertIn('enabled', user_ref)
        self.assertIn('password_expires_at', user_ref)

    @unit.skip_if_cache_disabled('identity')
    def test_cache_layer_get_user(self):
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        PROVIDERS.identity_api.create_user(user)
        ref = PROVIDERS.identity_api.get_user_by_name(user['name'], user['domain_id'])
        PROVIDERS.identity_api.get_user(ref['id'])
        domain_id, driver, entity_id = PROVIDERS.identity_api._get_domain_driver_and_entity_id(ref['id'])
        driver.delete_user(entity_id)
        self.assertDictEqual(ref, PROVIDERS.identity_api.get_user(ref['id']))
        PROVIDERS.identity_api.get_user.invalidate(PROVIDERS.identity_api, ref['id'])
        self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.get_user, ref['id'])
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        ref = PROVIDERS.identity_api.get_user_by_name(user['name'], user['domain_id'])
        user['description'] = uuid.uuid4().hex
        PROVIDERS.identity_api.get_user(ref['id'])
        user_updated = PROVIDERS.identity_api.update_user(ref['id'], user)
        self.assertLessEqual(PROVIDERS.identity_api.get_user(ref['id']).items(), user_updated.items())
        self.assertLessEqual(PROVIDERS.identity_api.get_user_by_name(ref['name'], ref['domain_id']).items(), user_updated.items())

    def test_get_user_returns_not_found(self):
        self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.get_user, uuid.uuid4().hex)

    def test_get_user_by_name(self):
        user_ref = PROVIDERS.identity_api.get_user_by_name(self.user_foo['name'], CONF.identity.default_domain_id)
        self.user_foo.pop('password')
        self.assertDictEqual(self.user_foo, user_ref)

    @unit.skip_if_cache_disabled('identity')
    def test_cache_layer_get_user_by_name(self):
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        PROVIDERS.identity_api.create_user(user)
        ref = PROVIDERS.identity_api.get_user_by_name(user['name'], user['domain_id'])
        domain_id, driver, entity_id = PROVIDERS.identity_api._get_domain_driver_and_entity_id(ref['id'])
        driver.delete_user(entity_id)
        self.assertDictEqual(ref, PROVIDERS.identity_api.get_user_by_name(user['name'], CONF.identity.default_domain_id))
        PROVIDERS.identity_api.get_user_by_name.invalidate(PROVIDERS.identity_api, user['name'], CONF.identity.default_domain_id)
        self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.get_user_by_name, user['name'], CONF.identity.default_domain_id)
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        ref = PROVIDERS.identity_api.get_user_by_name(user['name'], user['domain_id'])
        user['description'] = uuid.uuid4().hex
        user_updated = PROVIDERS.identity_api.update_user(ref['id'], user)
        self.assertLessEqual(PROVIDERS.identity_api.get_user(ref['id']).items(), user_updated.items())
        self.assertLessEqual(PROVIDERS.identity_api.get_user_by_name(ref['name'], ref['domain_id']).items(), user_updated.items())

    def test_get_user_by_name_returns_not_found(self):
        self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.get_user_by_name, uuid.uuid4().hex, CONF.identity.default_domain_id)

    def test_create_duplicate_user_name_fails(self):
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        self.assertRaises(exception.Conflict, PROVIDERS.identity_api.create_user, user)

    def test_create_duplicate_user_name_in_different_domains(self):
        new_domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
        user1 = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user2 = unit.new_user_ref(name=user1['name'], domain_id=new_domain['id'])
        PROVIDERS.identity_api.create_user(user1)
        PROVIDERS.identity_api.create_user(user2)

    def test_move_user_between_domains(self):
        domain1 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
        domain2 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
        user = unit.new_user_ref(domain_id=domain1['id'])
        user = PROVIDERS.identity_api.create_user(user)
        user['domain_id'] = domain2['id']
        self.assertRaises(exception.ValidationError, PROVIDERS.identity_api.update_user, user['id'], user)

    def test_rename_duplicate_user_name_fails(self):
        user1 = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user2 = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        PROVIDERS.identity_api.create_user(user1)
        user2 = PROVIDERS.identity_api.create_user(user2)
        user2['name'] = user1['name']
        self.assertRaises(exception.Conflict, PROVIDERS.identity_api.update_user, user2['id'], user2)

    def test_update_user_id_fails(self):
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        original_id = user['id']
        user['id'] = 'fake2'
        self.assertRaises(exception.ValidationError, PROVIDERS.identity_api.update_user, original_id, user)
        user_ref = PROVIDERS.identity_api.get_user(original_id)
        self.assertEqual(original_id, user_ref['id'])
        self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.get_user, 'fake2')

    def test_delete_user_with_group_project_domain_links(self):
        role1 = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role1['id'], role1)
        domain1 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
        project1 = unit.new_project_ref(domain_id=domain1['id'])
        PROVIDERS.resource_api.create_project(project1['id'], project1)
        user1 = unit.new_user_ref(domain_id=domain1['id'])
        user1 = PROVIDERS.identity_api.create_user(user1)
        group1 = unit.new_group_ref(domain_id=domain1['id'])
        group1 = PROVIDERS.identity_api.create_group(group1)
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=project1['id'], role_id=role1['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], domain_id=domain1['id'], role_id=role1['id'])
        PROVIDERS.identity_api.add_user_to_group(user_id=user1['id'], group_id=group1['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], project_id=project1['id'])
        self.assertEqual(1, len(roles_ref))
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], domain_id=domain1['id'])
        self.assertEqual(1, len(roles_ref))
        PROVIDERS.identity_api.check_user_in_group(user_id=user1['id'], group_id=group1['id'])
        PROVIDERS.identity_api.delete_user(user1['id'])
        self.assertRaises(exception.NotFound, PROVIDERS.identity_api.check_user_in_group, user1['id'], group1['id'])

    def test_delete_group_with_user_project_domain_links(self):
        role1 = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role1['id'], role1)
        domain1 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
        project1 = unit.new_project_ref(domain_id=domain1['id'])
        PROVIDERS.resource_api.create_project(project1['id'], project1)
        user1 = unit.new_user_ref(domain_id=domain1['id'])
        user1 = PROVIDERS.identity_api.create_user(user1)
        group1 = unit.new_group_ref(domain_id=domain1['id'])
        group1 = PROVIDERS.identity_api.create_group(group1)
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], project_id=project1['id'], role_id=role1['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], domain_id=domain1['id'], role_id=role1['id'])
        PROVIDERS.identity_api.add_user_to_group(user_id=user1['id'], group_id=group1['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], project_id=project1['id'])
        self.assertEqual(1, len(roles_ref))
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], domain_id=domain1['id'])
        self.assertEqual(1, len(roles_ref))
        PROVIDERS.identity_api.check_user_in_group(user_id=user1['id'], group_id=group1['id'])
        PROVIDERS.identity_api.delete_group(group1['id'])
        PROVIDERS.identity_api.get_user(user1['id'])

    def test_update_user_returns_not_found(self):
        user_id = uuid.uuid4().hex
        self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.update_user, user_id, {'id': user_id, 'domain_id': CONF.identity.default_domain_id})

    def test_delete_user_returns_not_found(self):
        self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.delete_user, uuid.uuid4().hex)

    def test_create_user_with_long_password(self):
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id, password='a' * 2000)
        PROVIDERS.identity_api.create_user(user)

    def test_create_user_missed_password(self):
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        PROVIDERS.identity_api.get_user(user['id'])
        with self.make_request():
            self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, user_id=user['id'], password='')
            self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, user_id=user['id'], password=None)

    def test_create_user_none_password(self):
        user = unit.new_user_ref(password=None, domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        PROVIDERS.identity_api.get_user(user['id'])
        with self.make_request():
            self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, user_id=user['id'], password='')
            self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, user_id=user['id'], password=None)

    def test_list_users(self):
        users = PROVIDERS.identity_api.list_users(domain_scope=self._set_domain_scope(CONF.identity.default_domain_id))
        self.assertEqual(len(default_fixtures.USERS), len(users))
        user_ids = set((user['id'] for user in users))
        expected_user_ids = set((getattr(self, 'user_%s' % user['name'])['id'] for user in default_fixtures.USERS))
        for user_ref in users:
            self.assertNotIn('password', user_ref)
        self.assertEqual(expected_user_ids, user_ids)

    def _build_hints(self, hints, filters, fed_dict):
        for key in filters:
            hints.add_filter(key, fed_dict[key], comparator='equals')
        return hints

    def _build_fed_resource(self):
        new_mapping = unit.new_mapping_ref()
        PROVIDERS.federation_api.create_mapping(new_mapping['id'], new_mapping)
        for idp_id, protocol_id in [('ORG_IDP', 'saml2'), ('myidp', 'mapped')]:
            new_idp = unit.new_identity_provider_ref(idp_id=idp_id, domain_id='default')
            new_protocol = unit.new_protocol_ref(protocol_id=protocol_id, idp_id=idp_id, mapping_id=new_mapping['id'])
            PROVIDERS.federation_api.create_idp(new_idp['id'], new_idp)
            PROVIDERS.federation_api.create_protocol(new_idp['id'], new_protocol['id'], new_protocol)

    def _test_list_users_with_attribute(self, filters, fed_dict):
        self._build_fed_resource()
        domain = self._get_domain_fixture()
        hints = driver_hints.Hints()
        hints = self._build_hints(hints, filters, fed_dict)
        users = PROVIDERS.identity_api.list_users(hints=hints)
        self.assertEqual(0, len(users))
        hints = self._build_hints(hints, filters, fed_dict)
        PROVIDERS.shadow_users_api.create_federated_user(domain['id'], fed_dict)
        users = PROVIDERS.identity_api.list_users(hints=hints)
        self.assertEqual(1, len(users))
        hints = self._build_hints(hints, filters, fed_dict)
        fed_dict2 = unit.new_federated_user_ref()
        fed_dict2['idp_id'] = 'myidp'
        fed_dict2['protocol_id'] = 'mapped'
        PROVIDERS.shadow_users_api.create_federated_user(domain['id'], fed_dict2)
        users = PROVIDERS.identity_api.list_users(hints=hints)
        self.assertEqual(1, len(users))
        hints = self._build_hints(hints, filters, fed_dict)
        if not any(('unique_id' in x['name'] for x in hints.filters)):
            hints = self._build_hints(hints, filters, fed_dict)
            fed_dict3 = unit.new_federated_user_ref()
            for filters_ in hints.filters:
                if filters_['name'] == 'idp_id':
                    fed_dict3['idp_id'] = fed_dict['idp_id']
                elif filters_['name'] == 'protocol_id':
                    fed_dict3['protocol_id'] = fed_dict['protocol_id']
            PROVIDERS.shadow_users_api.create_federated_user(domain['id'], fed_dict3)
            users = PROVIDERS.identity_api.list_users(hints=hints)
            self.assertEqual(2, len(users))

    def test_list_users_with_unique_id(self):
        federated_dict = unit.new_federated_user_ref()
        filters = ['unique_id']
        self._test_list_users_with_attribute(filters, federated_dict)

    def test_list_users_with_idp_id(self):
        federated_dict = unit.new_federated_user_ref()
        filters = ['idp_id']
        self._test_list_users_with_attribute(filters, federated_dict)

    def test_list_users_with_protocol_id(self):
        federated_dict = unit.new_federated_user_ref()
        filters = ['protocol_id']
        self._test_list_users_with_attribute(filters, federated_dict)

    def test_list_users_with_unique_id_and_idp_id(self):
        federated_dict = unit.new_federated_user_ref()
        filters = ['unique_id', 'idp_id']
        self._test_list_users_with_attribute(filters, federated_dict)

    def test_list_users_with_unique_id_and_protocol_id(self):
        federated_dict = unit.new_federated_user_ref()
        filters = ['unique_id', 'protocol_id']
        self._test_list_users_with_attribute(filters, federated_dict)

    def test_list_users_with_idp_id_protocol_id(self):
        federated_dict = unit.new_federated_user_ref()
        filters = ['idp_id', 'protocol_id']
        self._test_list_users_with_attribute(filters, federated_dict)

    def test_list_users_with_all_federated_attributes(self):
        federated_dict = unit.new_federated_user_ref()
        filters = ['unique_id', 'idp_id', 'protocol_id']
        self._test_list_users_with_attribute(filters, federated_dict)

    def test_list_users_with_name(self):
        self._build_fed_resource()
        federated_dict_1 = unit.new_federated_user_ref(display_name='test1@federation.org')
        federated_dict_2 = unit.new_federated_user_ref(display_name='test2@federation.org')
        domain = self._get_domain_fixture()
        hints = driver_hints.Hints()
        hints.add_filter('name', 'test1@federation.org')
        users = self.identity_api.list_users(hints=hints)
        self.assertEqual(0, len(users))
        self.shadow_users_api.create_federated_user(domain['id'], federated_dict_1)
        self.shadow_users_api.create_federated_user(domain['id'], federated_dict_2)
        hints = driver_hints.Hints()
        hints.add_filter('name', 'test1@federation.org')
        users = self.identity_api.list_users(hints=hints)
        self.assertEqual(1, len(users))
        hints = driver_hints.Hints()
        hints.add_filter('name', 'test1@federation.org')
        hints.add_filter('idp_id', 'ORG_IDP')
        users = self.identity_api.list_users(hints=hints)
        self.assertEqual(1, len(users))

    def test_list_groups(self):
        group1 = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group2 = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group1 = PROVIDERS.identity_api.create_group(group1)
        group2 = PROVIDERS.identity_api.create_group(group2)
        groups = PROVIDERS.identity_api.list_groups(domain_scope=self._set_domain_scope(CONF.identity.default_domain_id))
        self.assertEqual(2, len(groups))
        group_ids = []
        for group in groups:
            group_ids.append(group.get('id'))
        self.assertIn(group1['id'], group_ids)
        self.assertIn(group2['id'], group_ids)

    def test_create_user_doesnt_modify_passed_in_dict(self):
        new_user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        original_user = new_user.copy()
        PROVIDERS.identity_api.create_user(new_user)
        self.assertDictEqual(original_user, new_user)

    def test_update_user_enable(self):
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        user_ref = PROVIDERS.identity_api.get_user(user['id'])
        self.assertTrue(user_ref['enabled'])
        user['enabled'] = False
        PROVIDERS.identity_api.update_user(user['id'], user)
        user_ref = PROVIDERS.identity_api.get_user(user['id'])
        self.assertEqual(user['enabled'], user_ref['enabled'])
        del user['enabled']
        PROVIDERS.identity_api.update_user(user['id'], user)
        user_ref = PROVIDERS.identity_api.get_user(user['id'])
        self.assertFalse(user_ref['enabled'])
        user['enabled'] = True
        PROVIDERS.identity_api.update_user(user['id'], user)
        user_ref = PROVIDERS.identity_api.get_user(user['id'])
        self.assertEqual(user['enabled'], user_ref['enabled'])
        del user['enabled']
        PROVIDERS.identity_api.update_user(user['id'], user)
        user_ref = PROVIDERS.identity_api.get_user(user['id'])
        self.assertTrue(user_ref['enabled'])

    def test_update_user_name(self):
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        user_ref = PROVIDERS.identity_api.get_user(user['id'])
        self.assertEqual(user['name'], user_ref['name'])
        changed_name = user_ref['name'] + '_changed'
        user_ref['name'] = changed_name
        updated_user = PROVIDERS.identity_api.update_user(user_ref['id'], user_ref)
        updated_user.pop('extra', None)
        self.assertDictEqual(user_ref, updated_user)
        user_ref = PROVIDERS.identity_api.get_user(user_ref['id'])
        self.assertEqual(changed_name, user_ref['name'])

    def test_add_user_to_group(self):
        domain = self._get_domain_fixture()
        new_group = unit.new_group_ref(domain_id=domain['id'])
        new_group = PROVIDERS.identity_api.create_group(new_group)
        new_user = unit.new_user_ref(domain_id=domain['id'])
        new_user = PROVIDERS.identity_api.create_user(new_user)
        PROVIDERS.identity_api.add_user_to_group(new_user['id'], new_group['id'])
        groups = PROVIDERS.identity_api.list_groups_for_user(new_user['id'])
        found = False
        for x in groups:
            if x['id'] == new_group['id']:
                found = True
        self.assertTrue(found)

    def test_add_user_to_group_returns_not_found(self):
        domain = self._get_domain_fixture()
        new_user = unit.new_user_ref(domain_id=domain['id'])
        new_user = PROVIDERS.identity_api.create_user(new_user)
        self.assertRaises(exception.GroupNotFound, PROVIDERS.identity_api.add_user_to_group, new_user['id'], uuid.uuid4().hex)
        new_group = unit.new_group_ref(domain_id=domain['id'])
        new_group = PROVIDERS.identity_api.create_group(new_group)
        self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.add_user_to_group, uuid.uuid4().hex, new_group['id'])
        self.assertRaises(exception.NotFound, PROVIDERS.identity_api.add_user_to_group, uuid.uuid4().hex, uuid.uuid4().hex)

    def test_check_user_in_group(self):
        domain = self._get_domain_fixture()
        new_group = unit.new_group_ref(domain_id=domain['id'])
        new_group = PROVIDERS.identity_api.create_group(new_group)
        new_user = unit.new_user_ref(domain_id=domain['id'])
        new_user = PROVIDERS.identity_api.create_user(new_user)
        PROVIDERS.identity_api.add_user_to_group(new_user['id'], new_group['id'])
        PROVIDERS.identity_api.check_user_in_group(new_user['id'], new_group['id'])

    def test_check_user_not_in_group(self):
        new_group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        new_group = PROVIDERS.identity_api.create_group(new_group)
        new_user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        new_user = PROVIDERS.identity_api.create_user(new_user)
        self.assertRaises(exception.NotFound, PROVIDERS.identity_api.check_user_in_group, new_user['id'], new_group['id'])

    def test_check_user_in_group_returns_not_found(self):
        new_user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        new_user = PROVIDERS.identity_api.create_user(new_user)
        new_group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        new_group = PROVIDERS.identity_api.create_group(new_group)
        self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.check_user_in_group, uuid.uuid4().hex, new_group['id'])
        self.assertRaises(exception.GroupNotFound, PROVIDERS.identity_api.check_user_in_group, new_user['id'], uuid.uuid4().hex)
        self.assertRaises(exception.NotFound, PROVIDERS.identity_api.check_user_in_group, uuid.uuid4().hex, uuid.uuid4().hex)

    def test_list_users_in_group(self):
        domain = self._get_domain_fixture()
        new_group = unit.new_group_ref(domain_id=domain['id'])
        new_group = PROVIDERS.identity_api.create_group(new_group)
        user_refs = PROVIDERS.identity_api.list_users_in_group(new_group['id'])
        self.assertEqual([], user_refs)
        new_user = unit.new_user_ref(domain_id=domain['id'])
        new_user = PROVIDERS.identity_api.create_user(new_user)
        PROVIDERS.identity_api.add_user_to_group(new_user['id'], new_group['id'])
        user_refs = PROVIDERS.identity_api.list_users_in_group(new_group['id'])
        found = False
        for x in user_refs:
            if x['id'] == new_user['id']:
                found = True
            self.assertNotIn('password', x)
        self.assertTrue(found)

    def test_list_users_in_group_returns_not_found(self):
        self.assertRaises(exception.GroupNotFound, PROVIDERS.identity_api.list_users_in_group, uuid.uuid4().hex)

    def test_list_groups_for_user(self):
        domain = self._get_domain_fixture()
        test_groups = []
        test_users = []
        GROUP_COUNT = 3
        USER_COUNT = 2
        for x in range(0, USER_COUNT):
            new_user = unit.new_user_ref(domain_id=domain['id'])
            new_user = PROVIDERS.identity_api.create_user(new_user)
            test_users.append(new_user)
        positive_user = test_users[0]
        negative_user = test_users[1]
        for x in range(0, USER_COUNT):
            group_refs = PROVIDERS.identity_api.list_groups_for_user(test_users[x]['id'])
            self.assertEqual(0, len(group_refs))
        for x in range(0, GROUP_COUNT):
            before_count = x
            after_count = x + 1
            new_group = unit.new_group_ref(domain_id=domain['id'])
            new_group = PROVIDERS.identity_api.create_group(new_group)
            test_groups.append(new_group)
            group_refs = PROVIDERS.identity_api.list_groups_for_user(positive_user['id'])
            self.assertEqual(before_count, len(group_refs))
            PROVIDERS.identity_api.add_user_to_group(positive_user['id'], new_group['id'])
            group_refs = PROVIDERS.identity_api.list_groups_for_user(positive_user['id'])
            self.assertEqual(after_count, len(group_refs))
            group_refs = PROVIDERS.identity_api.list_groups_for_user(negative_user['id'])
            self.assertEqual(0, len(group_refs))

    def test_remove_user_from_group(self):
        domain = self._get_domain_fixture()
        new_group = unit.new_group_ref(domain_id=domain['id'])
        new_group = PROVIDERS.identity_api.create_group(new_group)
        new_user = unit.new_user_ref(domain_id=domain['id'])
        new_user = PROVIDERS.identity_api.create_user(new_user)
        PROVIDERS.identity_api.add_user_to_group(new_user['id'], new_group['id'])
        groups = PROVIDERS.identity_api.list_groups_for_user(new_user['id'])
        self.assertIn(new_group['id'], [x['id'] for x in groups])
        PROVIDERS.identity_api.remove_user_from_group(new_user['id'], new_group['id'])
        groups = PROVIDERS.identity_api.list_groups_for_user(new_user['id'])
        self.assertNotIn(new_group['id'], [x['id'] for x in groups])

    def test_remove_user_from_group_returns_not_found(self):
        domain = self._get_domain_fixture()
        new_user = unit.new_user_ref(domain_id=domain['id'])
        new_user = PROVIDERS.identity_api.create_user(new_user)
        new_group = unit.new_group_ref(domain_id=domain['id'])
        new_group = PROVIDERS.identity_api.create_group(new_group)
        self.assertRaises(exception.GroupNotFound, PROVIDERS.identity_api.remove_user_from_group, new_user['id'], uuid.uuid4().hex)
        self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.remove_user_from_group, uuid.uuid4().hex, new_group['id'])
        self.assertRaises(exception.NotFound, PROVIDERS.identity_api.remove_user_from_group, uuid.uuid4().hex, uuid.uuid4().hex)

    def test_group_crud(self):
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        group = unit.new_group_ref(domain_id=domain['id'])
        group = PROVIDERS.identity_api.create_group(group)
        group_ref = PROVIDERS.identity_api.get_group(group['id'])
        self.assertLessEqual(group.items(), group_ref.items())
        group['name'] = uuid.uuid4().hex
        PROVIDERS.identity_api.update_group(group['id'], group)
        group_ref = PROVIDERS.identity_api.get_group(group['id'])
        self.assertLessEqual(group.items(), group_ref.items())
        PROVIDERS.identity_api.delete_group(group['id'])
        self.assertRaises(exception.GroupNotFound, PROVIDERS.identity_api.get_group, group['id'])

    def test_create_group_name_with_trailing_whitespace(self):
        group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group_name = group['name'] = group['name'] + '    '
        group_returned = PROVIDERS.identity_api.create_group(group)
        self.assertEqual(group_returned['name'], group_name.strip())

    def test_update_group_name_with_trailing_whitespace(self):
        group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group_create = PROVIDERS.identity_api.create_group(group)
        group_name = group['name'] = group['name'] + '    '
        group_update = PROVIDERS.identity_api.update_group(group_create['id'], group)
        self.assertEqual(group_update['id'], group_create['id'])
        self.assertEqual(group_update['name'], group_name.strip())

    def test_get_group_by_name(self):
        group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group_name = group['name']
        group = PROVIDERS.identity_api.create_group(group)
        spoiler = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        PROVIDERS.identity_api.create_group(spoiler)
        group_ref = PROVIDERS.identity_api.get_group_by_name(group_name, CONF.identity.default_domain_id)
        self.assertDictEqual(group, group_ref)

    def test_get_group_by_name_returns_not_found(self):
        self.assertRaises(exception.GroupNotFound, PROVIDERS.identity_api.get_group_by_name, uuid.uuid4().hex, CONF.identity.default_domain_id)

    @unit.skip_if_cache_disabled('identity')
    def test_cache_layer_group_crud(self):
        group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group = PROVIDERS.identity_api.create_group(group)
        group_ref = PROVIDERS.identity_api.get_group(group['id'])
        domain_id, driver, entity_id = PROVIDERS.identity_api._get_domain_driver_and_entity_id(group['id'])
        driver.delete_group(entity_id)
        self.assertEqual(group_ref, PROVIDERS.identity_api.get_group(group['id']))
        PROVIDERS.identity_api.get_group.invalidate(PROVIDERS.identity_api, group['id'])
        self.assertRaises(exception.GroupNotFound, PROVIDERS.identity_api.get_group, group['id'])
        group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group = PROVIDERS.identity_api.create_group(group)
        PROVIDERS.identity_api.get_group(group['id'])
        group['name'] = uuid.uuid4().hex
        group_ref = PROVIDERS.identity_api.update_group(group['id'], group)
        self.assertLessEqual(PROVIDERS.identity_api.get_group(group['id']).items(), group_ref.items())

    def test_create_duplicate_group_name_fails(self):
        group1 = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group2 = unit.new_group_ref(domain_id=CONF.identity.default_domain_id, name=group1['name'])
        group1 = PROVIDERS.identity_api.create_group(group1)
        self.assertRaises(exception.Conflict, PROVIDERS.identity_api.create_group, group2)

    def test_create_duplicate_group_name_in_different_domains(self):
        new_domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
        group1 = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group2 = unit.new_group_ref(domain_id=new_domain['id'], name=group1['name'])
        group1 = PROVIDERS.identity_api.create_group(group1)
        group2 = PROVIDERS.identity_api.create_group(group2)

    def test_move_group_between_domains(self):
        domain1 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
        domain2 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
        group = unit.new_group_ref(domain_id=domain1['id'])
        group = PROVIDERS.identity_api.create_group(group)
        group['domain_id'] = domain2['id']
        self.assertRaises(exception.ValidationError, PROVIDERS.identity_api.update_group, group['id'], group)

    def test_user_crud(self):
        user_dict = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        del user_dict['id']
        user = PROVIDERS.identity_api.create_user(user_dict)
        user_ref = PROVIDERS.identity_api.get_user(user['id'])
        del user_dict['password']
        user_ref_dict = {x: user_ref[x] for x in user_ref}
        self.assertLessEqual(user_dict.items(), user_ref_dict.items())
        user_dict['password'] = uuid.uuid4().hex
        PROVIDERS.identity_api.update_user(user['id'], user_dict)
        user_ref = PROVIDERS.identity_api.get_user(user['id'])
        del user_dict['password']
        user_ref_dict = {x: user_ref[x] for x in user_ref}
        self.assertLessEqual(user_dict.items(), user_ref_dict.items())
        PROVIDERS.identity_api.delete_user(user['id'])
        self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.get_user, user['id'])

    def test_arbitrary_attributes_are_returned_from_create_user(self):
        attr_value = uuid.uuid4().hex
        user_data = unit.new_user_ref(domain_id=CONF.identity.default_domain_id, arbitrary_attr=attr_value)
        user = PROVIDERS.identity_api.create_user(user_data)
        self.assertEqual(attr_value, user['arbitrary_attr'])

    def test_arbitrary_attributes_are_returned_from_get_user(self):
        attr_value = uuid.uuid4().hex
        user_data = unit.new_user_ref(domain_id=CONF.identity.default_domain_id, arbitrary_attr=attr_value)
        user_data = PROVIDERS.identity_api.create_user(user_data)
        user = PROVIDERS.identity_api.get_user(user_data['id'])
        self.assertEqual(attr_value, user['arbitrary_attr'])

    def test_new_arbitrary_attributes_are_returned_from_update_user(self):
        user_data = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user_data)
        attr_value = uuid.uuid4().hex
        user['arbitrary_attr'] = attr_value
        updated_user = PROVIDERS.identity_api.update_user(user['id'], user)
        self.assertEqual(attr_value, updated_user['arbitrary_attr'])

    def test_updated_arbitrary_attributes_are_returned_from_update_user(self):
        attr_value = uuid.uuid4().hex
        user_data = unit.new_user_ref(domain_id=CONF.identity.default_domain_id, arbitrary_attr=attr_value)
        new_attr_value = uuid.uuid4().hex
        user = PROVIDERS.identity_api.create_user(user_data)
        user['arbitrary_attr'] = new_attr_value
        updated_user = PROVIDERS.identity_api.update_user(user['id'], user)
        self.assertEqual(new_attr_value, updated_user['arbitrary_attr'])

    def test_user_update_and_user_get_return_same_response(self):
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        updated_user = {'enabled': False}
        updated_user_ref = PROVIDERS.identity_api.update_user(user['id'], updated_user)
        updated_user_ref.pop('extra', None)
        self.assertIs(False, updated_user_ref['enabled'])
        user_ref = PROVIDERS.identity_api.get_user(user['id'])
        self.assertDictEqual(updated_user_ref, user_ref)

    @unit.skip_if_no_multiple_domains_support
    def test_list_domains_filtered_and_limited(self):

        def create_domains(domain_count, domain_name_prefix):
            for _ in range(domain_count):
                domain_name = '%s-%s' % (domain_name_prefix, uuid.uuid4().hex)
                domain = unit.new_domain_ref(name=domain_name)
                self.domain_list[domain_name] = PROVIDERS.resource_api.create_domain(domain['id'], domain)

        def clean_up_domains():
            for _, domain in self.domain_list.items():
                domain['enabled'] = False
                PROVIDERS.resource_api.update_domain(domain['id'], domain)
                PROVIDERS.resource_api.delete_domain(domain['id'])
        self.domain_list = {}
        create_domains(2, 'domaingroup1')
        create_domains(3, 'domaingroup2')
        self.addCleanup(clean_up_domains)
        unfiltered_domains = PROVIDERS.resource_api.list_domains()
        self.config_fixture.config(list_limit=4)
        hints = driver_hints.Hints()
        entities = PROVIDERS.resource_api.list_domains(hints=hints)
        self.assertThat(entities, matchers.HasLength(hints.limit['limit']))
        self.assertTrue(hints.limit['truncated'])
        hints = driver_hints.Hints()
        hints.add_filter('name', unfiltered_domains[3]['name'])
        entities = PROVIDERS.resource_api.list_domains(hints=hints)
        self.assertThat(entities, matchers.HasLength(1))
        self.assertEqual(entities[0], unfiltered_domains[3])
        hints = driver_hints.Hints()
        hints.add_filter('name', 'domaingroup1', comparator='startswith')
        entities = PROVIDERS.resource_api.list_domains(hints=hints)
        self.assertThat(entities, matchers.HasLength(2))
        self.assertThat(entities[0]['name'], matchers.StartsWith('domaingroup1'))
        self.assertThat(entities[1]['name'], matchers.StartsWith('domaingroup1'))

    @unit.skip_if_no_multiple_domains_support
    def test_list_limit_for_domains(self):

        def create_domains(count):
            for _ in range(count):
                domain = unit.new_domain_ref()
                self.domain_list.append(PROVIDERS.resource_api.create_domain(domain['id'], domain))

        def clean_up_domains():
            for domain in self.domain_list:
                PROVIDERS.resource_api.update_domain(domain['id'], {'enabled': False})
                PROVIDERS.resource_api.delete_domain(domain['id'])
        self.domain_list = []
        create_domains(6)
        self.addCleanup(clean_up_domains)
        for x in range(1, 7):
            self.config_fixture.config(group='resource', list_limit=x)
            hints = driver_hints.Hints()
            entities = PROVIDERS.resource_api.list_domains(hints=hints)
            self.assertThat(entities, matchers.HasLength(hints.limit['limit']))