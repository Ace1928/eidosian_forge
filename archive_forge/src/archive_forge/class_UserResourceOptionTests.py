import datetime
import uuid
import freezegun
import passlib.hash
from keystone.common import password_hashing
from keystone.common import provider_api
from keystone.common import resource_options
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.identity.backends import base
from keystone.identity.backends import resource_options as iro
from keystone.identity.backends import sql_model as model
from keystone.tests.unit import test_backend_sql
class UserResourceOptionTests(test_backend_sql.SqlTests):

    def setUp(self):
        super(UserResourceOptionTests, self).setUp()
        self.addCleanup(iro.register_user_options)
        self.addCleanup(iro.USER_OPTIONS_REGISTRY._registered_options.clear)
        self.option1 = resource_options.ResourceOption('opt1', 'option1')
        self.option2 = resource_options.ResourceOption('opt2', 'option2')
        self.cleanup_instance('option1', 'option2')
        iro.USER_OPTIONS_REGISTRY._registered_options.clear()
        iro.USER_OPTIONS_REGISTRY.register_option(self.option1)
        iro.USER_OPTIONS_REGISTRY.register_option(self.option2)

    def test_user_set_option_in_resource_option(self):
        user = self._create_user(self._get_user_dict())
        opt_value = uuid.uuid4().hex
        user['options'][self.option1.option_name] = opt_value
        new_ref = PROVIDERS.identity_api.update_user(user['id'], user)
        self.assertEqual(opt_value, new_ref['options'][self.option1.option_name])
        raw_ref = self._get_user_ref(user['id'])
        self.assertIn(self.option1.option_id, raw_ref._resource_option_mapper)
        self.assertEqual(opt_value, raw_ref._resource_option_mapper[self.option1.option_id].option_value)
        api_get_ref = PROVIDERS.identity_api.get_user(user['id'])
        self.assertEqual(opt_value, api_get_ref['options'][self.option1.option_name])

    def test_user_add_update_delete_option_in_resource_option(self):
        user = self._create_user(self._get_user_dict())
        opt_value = uuid.uuid4().hex
        new_opt_value = uuid.uuid4().hex
        user['options'][self.option1.option_name] = opt_value
        new_ref = PROVIDERS.identity_api.update_user(user['id'], user)
        self.assertEqual(opt_value, new_ref['options'][self.option1.option_name])
        user['options'][self.option1.option_name] = new_opt_value
        new_ref = PROVIDERS.identity_api.update_user(user['id'], user)
        self.assertEqual(new_opt_value, new_ref['options'][self.option1.option_name])
        user['options'][self.option1.option_name] = None
        new_ref = PROVIDERS.identity_api.update_user(user['id'], user)
        self.assertNotIn(self.option1.option_name, new_ref['options'])

    def test_user_add_delete_resource_option_existing_option_values(self):
        user = self._create_user(self._get_user_dict())
        opt_value = uuid.uuid4().hex
        opt2_value = uuid.uuid4().hex
        user['options'][self.option1.option_name] = opt_value
        new_ref = PROVIDERS.identity_api.update_user(user['id'], user)
        self.assertEqual(opt_value, new_ref['options'][self.option1.option_name])
        del user['options'][self.option1.option_name]
        user['options'][self.option2.option_name] = opt2_value
        new_ref = PROVIDERS.identity_api.update_user(user['id'], user)
        self.assertEqual(opt_value, new_ref['options'][self.option1.option_name])
        self.assertEqual(opt2_value, new_ref['options'][self.option2.option_name])
        raw_ref = self._get_user_ref(user['id'])
        self.assertEqual(opt_value, raw_ref._resource_option_mapper[self.option1.option_id].option_value)
        self.assertEqual(opt2_value, raw_ref._resource_option_mapper[self.option2.option_id].option_value)
        user['options'][self.option1.option_name] = None
        new_ref = PROVIDERS.identity_api.update_user(user['id'], user)
        self.assertNotIn(self.option1.option_name, new_ref['options'])
        self.assertEqual(opt2_value, new_ref['options'][self.option2.option_name])
        raw_ref = self._get_user_ref(user['id'])
        self.assertNotIn(raw_ref._resource_option_mapper, self.option1.option_id)
        self.assertEqual(opt2_value, raw_ref._resource_option_mapper[self.option2.option_id].option_value)

    def test_unregistered_resource_option_deleted(self):
        user = self._create_user(self._get_user_dict())
        opt_value = uuid.uuid4().hex
        opt2_value = uuid.uuid4().hex
        user['options'][self.option1.option_name] = opt_value
        new_ref = PROVIDERS.identity_api.update_user(user['id'], user)
        self.assertEqual(opt_value, new_ref['options'][self.option1.option_name])
        del user['options'][self.option1.option_name]
        user['options'][self.option2.option_name] = opt2_value
        new_ref = PROVIDERS.identity_api.update_user(user['id'], user)
        self.assertEqual(opt_value, new_ref['options'][self.option1.option_name])
        self.assertEqual(opt2_value, new_ref['options'][self.option2.option_name])
        raw_ref = self._get_user_ref(user['id'])
        self.assertEqual(opt_value, raw_ref._resource_option_mapper[self.option1.option_id].option_value)
        self.assertEqual(opt2_value, raw_ref._resource_option_mapper[self.option2.option_id].option_value)
        iro.USER_OPTIONS_REGISTRY._registered_options.clear()
        iro.USER_OPTIONS_REGISTRY.register_option(self.option1)
        user['name'] = uuid.uuid4().hex
        new_ref = PROVIDERS.identity_api.update_user(user['id'], user)
        self.assertNotIn(self.option2.option_name, new_ref['options'])
        self.assertEqual(opt_value, new_ref['options'][self.option1.option_name])
        raw_ref = self._get_user_ref(user['id'])
        self.assertNotIn(raw_ref._resource_option_mapper, self.option2.option_id)
        self.assertEqual(opt_value, raw_ref._resource_option_mapper[self.option1.option_id].option_value)

    def _get_user_ref(self, user_id):
        with sql.session_for_read() as session:
            return session.get(model.User, user_id)

    def _create_user(self, user_dict):
        user_dict['id'] = uuid.uuid4().hex
        with sql.session_for_write() as session:
            user_ref = model.User.from_dict(user_dict)
            session.add(user_ref)
            return base.filter_user(user_ref.to_dict())

    def _get_user_dict(self):
        user = {'name': uuid.uuid4().hex, 'domain_id': CONF.identity.default_domain_id, 'enabled': True, 'password': uuid.uuid4().hex}
        return user