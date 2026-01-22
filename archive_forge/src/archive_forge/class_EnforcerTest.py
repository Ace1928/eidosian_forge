import os
from unittest import mock
import yaml
import fixtures
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslotest import base as test_base
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy import policy
from oslo_policy.tests import base
class EnforcerTest(base.PolicyBaseTestCase):

    def setUp(self):
        super(EnforcerTest, self).setUp()
        self.create_config_file('policy.json', POLICY_JSON_CONTENTS)

    def _test_scenario_with_opts_registered(self, scenario, *args, **kwargs):
        self.enforcer.register_default(policy.RuleDefault(name='admin', check_str='is_admin:False'))
        self.enforcer.register_default(policy.RuleDefault(name='owner', check_str='role:owner'))
        scenario(*args, **kwargs)
        self.assertIn('owner', self.enforcer.rules)
        self.assertEqual('role:owner', str(self.enforcer.rules['owner']))
        self.assertEqual('is_admin:True', str(self.enforcer.rules['admin']))
        self.assertIn('owner', self.enforcer.registered_rules)
        self.assertIn('admin', self.enforcer.registered_rules)
        self.assertNotIn('default', self.enforcer.registered_rules)
        self.assertNotIn('owner', self.enforcer.file_rules)
        self.assertIn('admin', self.enforcer.file_rules)
        self.assertIn('default', self.enforcer.file_rules)

    def test_load_file(self):
        self.conf.set_override('policy_dirs', [], group='oslo_policy')
        self.enforcer.load_rules(True)
        self.assertIsNotNone(self.enforcer.rules)
        self.assertIn('default', self.enforcer.rules)
        self.assertIn('admin', self.enforcer.rules)
        self.assertEqual('is_admin:True', str(self.enforcer.rules['admin']))

    def test_load_file_opts_registered(self):
        self._test_scenario_with_opts_registered(self.test_load_file)

    def test_load_directory(self):
        self.create_config_file(os.path.join('policy.d', 'a.conf'), POLICY_A_CONTENTS)
        self.create_config_file(os.path.join('policy.d', 'b.conf'), POLICY_B_CONTENTS)
        self.enforcer.load_rules(True)
        self.assertIsNotNone(self.enforcer.rules)
        loaded_rules = jsonutils.loads(str(self.enforcer.rules))
        self.assertEqual('role:fakeB', loaded_rules['default'])
        self.assertEqual('is_admin:True', loaded_rules['admin'])

    def test_load_directory_after_file_update(self):
        self.create_config_file(os.path.join('policy.d', 'a.conf'), POLICY_A_CONTENTS)
        self.enforcer.load_rules(True)
        self.assertIsNotNone(self.enforcer.rules)
        loaded_rules = jsonutils.loads(str(self.enforcer.rules))
        self.assertEqual('role:fakeA', loaded_rules['default'])
        self.assertEqual('is_admin:True', loaded_rules['admin'])
        new_policy_json_contents = jsonutils.dumps({'default': 'rule:admin', 'admin': 'is_admin:True', 'foo': 'rule:bar'})
        self.create_config_file('policy.json', new_policy_json_contents)
        policy_file_path = self.get_config_file_fullname('policy.json')
        stinfo = os.stat(policy_file_path)
        os.utime(policy_file_path, (stinfo.st_atime + 42, stinfo.st_mtime + 42))
        self.enforcer.load_rules()
        self.assertIsNotNone(self.enforcer.rules)
        loaded_rules = jsonutils.loads(str(self.enforcer.rules))
        self.assertEqual('role:fakeA', loaded_rules['default'])
        self.assertEqual('is_admin:True', loaded_rules['admin'])
        self.assertEqual('rule:bar', loaded_rules['foo'])

    def test_load_directory_after_file_is_emptied(self):

        def dict_rules(enforcer_rules):
            """Converts enforcer rules to dictionary.

            :param enforcer_rules: enforcer rules represented as a class Rules
            :return: enforcer rules represented as a dictionary
            """
            return jsonutils.loads(str(enforcer_rules))
        self.assertEqual(self.enforcer.rules, {})
        self.enforcer.load_rules()
        main_policy_file_rules = jsonutils.loads(POLICY_JSON_CONTENTS)
        self.assertEqual(main_policy_file_rules, dict_rules(self.enforcer.rules))
        folder_policy_file = os.path.join('policy.d', 'a.conf')
        self.create_config_file(folder_policy_file, POLICY_A_CONTENTS)
        self.enforcer.load_rules()
        expected_rules = main_policy_file_rules.copy()
        expected_rules.update(jsonutils.loads(POLICY_A_CONTENTS))
        self.assertEqual(expected_rules, dict_rules(self.enforcer.rules))
        self.create_config_file(folder_policy_file, '{}')
        absolute_folder_policy_file_path = self.get_config_file_fullname(folder_policy_file)
        stinfo = os.stat(absolute_folder_policy_file_path)
        os.utime(absolute_folder_policy_file_path, (stinfo.st_atime + 42, stinfo.st_mtime + 42))
        self.enforcer.load_rules()
        self.assertEqual(main_policy_file_rules, dict_rules(self.enforcer.rules))

    def test_load_directory_opts_registered(self):
        self._test_scenario_with_opts_registered(self.test_load_directory)

    def test_load_directory_caching_with_files_updated(self):
        self.create_config_file(os.path.join('policy.d', 'a.conf'), POLICY_A_CONTENTS)
        self.enforcer.load_rules(False)
        self.assertIsNotNone(self.enforcer.rules)
        old = next(iter(self.enforcer._policy_dir_mtimes))
        self.assertEqual(1, len(self.enforcer._policy_dir_mtimes))
        conf_path = os.path.join(self.config_dir, os.path.join('policy.d', 'a.conf'))
        stinfo = os.stat(conf_path)
        os.utime(conf_path, (stinfo.st_atime + 10, stinfo.st_mtime + 10))
        self.enforcer.load_rules(False)
        self.assertEqual(1, len(self.enforcer._policy_dir_mtimes))
        self.assertEqual(old, next(iter(self.enforcer._policy_dir_mtimes)))
        loaded_rules = jsonutils.loads(str(self.enforcer.rules))
        self.assertEqual('is_admin:True', loaded_rules['admin'])

    def test_load_directory_caching_with_files_updated_opts_registered(self):
        self._test_scenario_with_opts_registered(self.test_load_directory_caching_with_files_updated)

    def test_load_directory_caching_with_files_same(self, overwrite=True):
        self.enforcer.overwrite = overwrite
        self.create_config_file(os.path.join('policy.d', 'a.conf'), POLICY_A_CONTENTS)
        self.enforcer.load_rules(False)
        self.assertIsNotNone(self.enforcer.rules)
        old = next(iter(self.enforcer._policy_dir_mtimes))
        self.assertEqual(1, len(self.enforcer._policy_dir_mtimes))
        self.enforcer.load_rules(False)
        self.assertEqual(1, len(self.enforcer._policy_dir_mtimes))
        self.assertEqual(old, next(iter(self.enforcer._policy_dir_mtimes)))
        loaded_rules = jsonutils.loads(str(self.enforcer.rules))
        self.assertEqual('is_admin:True', loaded_rules['admin'])

    def test_load_directory_caching_with_files_same_but_overwrite_false(self):
        self.test_load_directory_caching_with_files_same(overwrite=False)

    def test_load_directory_caching_with_files_same_opts_registered(self):
        self._test_scenario_with_opts_registered(self.test_load_directory_caching_with_files_same)

    def test_load_dir_caching_with_files_same_overwrite_false_opts_reg(self):
        test = getattr(self, 'test_load_directory_caching_with_files_same_but_overwrite_false')
        self._test_scenario_with_opts_registered(test)

    @mock.patch.object(policy, 'LOG')
    def test_load_json_file_log_warning(self, mock_log):
        rules = jsonutils.dumps({'foo': 'rule:bar'})
        self.create_config_file('policy.json', rules)
        self.enforcer.load_rules(True)
        mock_log.warning.assert_any_call(policy.WARN_JSON)

    @mock.patch.object(policy, 'LOG')
    def test_warning_on_redundant_file_rules(self, mock_log):
        rules = yaml.dump({'admin': 'is_admin:True'})
        self.create_config_file('policy.yaml', rules)
        path = self.get_config_file_fullname('policy.yaml')
        enforcer = policy.Enforcer(self.conf, policy_file=path)
        enforcer.register_default(policy.RuleDefault(name='admin', check_str='is_admin:True'))
        enforcer.load_rules(True)
        warn_msg = 'Policy Rules %(names)s specified in policy files are the same as the defaults provided by the service. You can remove these rules from policy files which will make maintenance easier. You can detect these redundant rules by ``oslopolicy-list-redundant`` tool also.'
        mock_log.warning.assert_any_call(warn_msg, {'names': ['admin']})

    def test_load_multiple_directories(self):
        self.create_config_file(os.path.join('policy.d', 'a.conf'), POLICY_A_CONTENTS)
        self.create_config_file(os.path.join('policy.d', 'b.conf'), POLICY_B_CONTENTS)
        self.create_config_file(os.path.join('policy.2.d', 'fake.conf'), POLICY_FAKE_CONTENTS)
        self.conf.set_override('policy_dirs', ['policy.d', 'policy.2.d'], group='oslo_policy')
        self.enforcer.load_rules(True)
        self.assertIsNotNone(self.enforcer.rules)
        loaded_rules = jsonutils.loads(str(self.enforcer.rules))
        self.assertEqual('role:fakeC', loaded_rules['default'])
        self.assertEqual('is_admin:True', loaded_rules['admin'])

    def test_load_multiple_directories_opts_registered(self):
        self._test_scenario_with_opts_registered(self.test_load_multiple_directories)

    def test_load_non_existed_directory(self):
        self.create_config_file(os.path.join('policy.d', 'a.conf'), POLICY_A_CONTENTS)
        self.conf.set_override('policy_dirs', ['policy.d', 'policy.x.d'], group='oslo_policy')
        self.enforcer.load_rules(True)
        self.assertIsNotNone(self.enforcer.rules)
        self.assertIn('default', self.enforcer.rules)
        self.assertIn('admin', self.enforcer.rules)

    def test_load_non_existed_directory_opts_registered(self):
        self._test_scenario_with_opts_registered(self.test_load_non_existed_directory)

    def test_load_policy_dirs_with_non_directory(self):
        self.create_config_file(os.path.join('policy.d', 'a.conf'), POLICY_A_CONTENTS)
        self.conf.set_override('policy_dirs', [os.path.join('policy.d', 'a.conf')], group='oslo_policy')
        self.assertRaises(ValueError, self.enforcer.load_rules, True)
        self.assertRaises(ValueError, self.enforcer.load_rules, False)

    @mock.patch('oslo_policy.policy.Enforcer.check_rules')
    def test_load_rules_twice(self, mock_check_rules):
        self.enforcer.load_rules()
        self.enforcer.load_rules()
        self.assertEqual(1, mock_check_rules.call_count)

    @mock.patch('oslo_policy.policy.Enforcer.check_rules')
    def test_load_rules_twice_force(self, mock_check_rules):
        self.enforcer.load_rules(True)
        self.enforcer.load_rules(True)
        self.assertEqual(2, mock_check_rules.call_count)

    @mock.patch('oslo_policy.policy.Enforcer.check_rules')
    def test_load_rules_twice_clear(self, mock_check_rules):
        self.enforcer.load_rules()
        self.enforcer.clear()
        self.enforcer.load_rules(True)
        self.assertEqual(2, mock_check_rules.call_count)

    @mock.patch('oslo_policy.policy.Enforcer.check_rules')
    def test_load_directory_twice(self, mock_check_rules):
        self.create_config_file(os.path.join('policy.d', 'a.conf'), POLICY_A_CONTENTS)
        self.create_config_file(os.path.join('policy.d', 'b.conf'), POLICY_B_CONTENTS)
        self.enforcer.load_rules()
        self.enforcer.load_rules()
        self.assertEqual(1, mock_check_rules.call_count)
        self.assertIsNotNone(self.enforcer.rules)

    @mock.patch('oslo_policy.policy.Enforcer.check_rules')
    def test_load_directory_twice_force(self, mock_check_rules):
        self.create_config_file(os.path.join('policy.d', 'a.conf'), POLICY_A_CONTENTS)
        self.create_config_file(os.path.join('policy.d', 'b.conf'), POLICY_B_CONTENTS)
        self.enforcer.load_rules(True)
        self.enforcer.load_rules(True)
        self.assertEqual(2, mock_check_rules.call_count)
        self.assertIsNotNone(self.enforcer.rules)

    @mock.patch('oslo_policy.policy.Enforcer.check_rules')
    def test_load_directory_twice_changed(self, mock_check_rules):
        self.create_config_file(os.path.join('policy.d', 'a.conf'), POLICY_A_CONTENTS)
        self.enforcer.load_rules()
        conf_path = os.path.join(self.config_dir, os.path.join('policy.d', 'a.conf'))
        stinfo = os.stat(conf_path)
        os.utime(conf_path, (stinfo.st_atime + 10, stinfo.st_mtime + 10))
        self.enforcer.load_rules()
        self.assertEqual(2, mock_check_rules.call_count)
        self.assertIsNotNone(self.enforcer.rules)

    def test_set_rules_type(self):
        self.assertRaises(TypeError, self.enforcer.set_rules, 'dummy')

    @mock.patch.object(_cache_handler, 'delete_cached_file', mock.Mock())
    def test_clear(self):
        self.enforcer.rules = 'spam'
        self.enforcer.clear()
        self.assertEqual({}, self.enforcer.rules)
        self.assertIsNone(self.enforcer.default_rule)
        self.assertIsNone(self.enforcer.policy_path)

    def test_clear_opts_registered(self):
        self.enforcer.register_default(policy.RuleDefault(name='admin', check_str='is_admin:False'))
        self.enforcer.register_default(policy.RuleDefault(name='owner', check_str='role:owner'))
        self.test_clear()
        self.assertEqual({}, self.enforcer.registered_rules)

    def test_rule_with_check(self):
        rules_json = jsonutils.dumps({'deny_stack_user': 'not role:stack_user', 'cloudwatch:PutMetricData': ''})
        rules = policy.Rules.load(rules_json)
        self.enforcer.set_rules(rules)
        action = 'cloudwatch:PutMetricData'
        creds = {'roles': ''}
        self.assertTrue(self.enforcer.enforce(action, {}, creds))

    def test_enforcer_with_default_rule(self):
        rules_json = jsonutils.dumps({'deny_stack_user': 'not role:stack_user', 'cloudwatch:PutMetricData': ''})
        rules = policy.Rules.load(rules_json)
        default_rule = _checks.TrueCheck()
        enforcer = policy.Enforcer(self.conf, default_rule=default_rule)
        enforcer.set_rules(rules)
        action = 'cloudwatch:PutMetricData'
        creds = {'roles': ''}
        self.assertTrue(enforcer.enforce(action, {}, creds))

    def test_enforcer_force_reload_with_overwrite(self, opts_registered=0):
        self.create_config_file(os.path.join('policy.d', 'a.conf'), POLICY_A_CONTENTS)
        self.create_config_file(os.path.join('policy.d', 'b.conf'), POLICY_B_CONTENTS)
        self.enforcer.set_rules({'test': _parser.parse_rule('role:test')}, use_conf=True)
        self.enforcer.set_rules({'default': _parser.parse_rule('role:fakeZ')}, overwrite=False, use_conf=True)
        self.enforcer.overwrite = True
        self.assertFalse(self.enforcer.enforce('test', {}, {'roles': ['test']}))
        self.assertTrue(self.enforcer.enforce('default', {}, {'roles': ['fakeB']}))
        self.assertNotIn('test', self.enforcer.rules)
        self.assertIn('default', self.enforcer.rules)
        self.assertIn('admin', self.enforcer.rules)
        loaded_rules = jsonutils.loads(str(self.enforcer.rules))
        self.assertEqual(2 + opts_registered, len(loaded_rules))
        self.assertIn('role:fakeB', loaded_rules['default'])
        self.assertIn('is_admin:True', loaded_rules['admin'])

    def test_enforcer_force_reload_with_overwrite_opts_registered(self):
        self._test_scenario_with_opts_registered(self.test_enforcer_force_reload_with_overwrite, opts_registered=1)

    def test_enforcer_force_reload_without_overwrite(self, opts_registered=0):
        self.create_config_file(os.path.join('policy.d', 'a.conf'), POLICY_A_CONTENTS)
        self.create_config_file(os.path.join('policy.d', 'b.conf'), POLICY_B_CONTENTS)
        self.enforcer.set_rules({'test': _parser.parse_rule('role:test')}, use_conf=True)
        self.enforcer.set_rules({'default': _parser.parse_rule('role:fakeZ')}, overwrite=False, use_conf=True)
        self.enforcer.overwrite = False
        self.enforcer._is_directory_updated = lambda x, y: True
        self.assertTrue(self.enforcer.enforce('test', {}, {'roles': ['test']}))
        self.assertFalse(self.enforcer.enforce('default', {}, {'roles': ['fakeZ']}))
        self.assertIn('test', self.enforcer.rules)
        self.assertIn('default', self.enforcer.rules)
        self.assertIn('admin', self.enforcer.rules)
        loaded_rules = jsonutils.loads(str(self.enforcer.rules))
        self.assertEqual(3 + opts_registered, len(loaded_rules))
        self.assertIn('role:test', loaded_rules['test'])
        self.assertIn('role:fakeB', loaded_rules['default'])
        self.assertIn('is_admin:True', loaded_rules['admin'])

    def test_enforcer_force_reload_without_overwrite_opts_registered(self):
        self._test_scenario_with_opts_registered(self.test_enforcer_force_reload_without_overwrite, opts_registered=1)

    def test_enforcer_keep_use_conf_flag_after_reload(self):
        self.create_config_file(os.path.join('policy.d', 'a.conf'), POLICY_A_CONTENTS)
        self.create_config_file(os.path.join('policy.d', 'b.conf'), POLICY_B_CONTENTS)
        self.assertTrue(self.enforcer.use_conf)
        self.assertTrue(self.enforcer.enforce('default', {}, {'roles': ['fakeB']}))
        self.assertFalse(self.enforcer.enforce('test', {}, {'roles': ['test']}))
        self.assertTrue(self.enforcer.use_conf)
        self.assertFalse(self.enforcer.enforce('_dynamic_test_rule', {}, {'roles': ['test']}))
        rules = jsonutils.loads(str(self.enforcer.rules))
        rules['_dynamic_test_rule'] = 'role:test'
        with open(self.enforcer.policy_path, 'w') as f:
            f.write(jsonutils.dumps(rules))
        self.enforcer.load_rules(force_reload=True)
        self.assertTrue(self.enforcer.enforce('_dynamic_test_rule', {}, {'roles': ['test']}))

    def test_enforcer_keep_use_conf_flag_after_reload_opts_registered(self):
        self.enforcer.register_default(policy.RuleDefault(name='admin', check_str='is_admin:False'))
        self.enforcer.register_default(policy.RuleDefault(name='owner', check_str='role:owner'))
        self.test_enforcer_keep_use_conf_flag_after_reload()
        self.assertIn('owner', self.enforcer.rules)
        self.assertEqual('role:owner', str(self.enforcer.rules['owner']))
        self.assertEqual('is_admin:True', str(self.enforcer.rules['admin']))

    def test_enforcer_force_reload_false(self):
        self.enforcer.set_rules({'test': 'test'})
        self.enforcer.load_rules(force_reload=False)
        self.assertIn('test', self.enforcer.rules)
        self.assertNotIn('default', self.enforcer.rules)
        self.assertNotIn('admin', self.enforcer.rules)

    def test_enforcer_overwrite_rules(self):
        self.enforcer.set_rules({'test': 'test'})
        self.enforcer.set_rules({'test': 'test1'}, overwrite=True)
        self.assertEqual({'test': 'test1'}, self.enforcer.rules)

    def test_enforcer_update_rules(self):
        self.enforcer.set_rules({'test': 'test'})
        self.enforcer.set_rules({'test1': 'test1'}, overwrite=False)
        self.assertEqual({'test': 'test', 'test1': 'test1'}, self.enforcer.rules)

    def test_enforcer_with_default_policy_file(self):
        enforcer = policy.Enforcer(self.conf)
        self.assertEqual(self.conf.oslo_policy.policy_file, enforcer.policy_file)

    def test_enforcer_with_policy_file(self):
        enforcer = policy.Enforcer(self.conf, policy_file='non-default.json')
        self.assertEqual('non-default.json', enforcer.policy_file)

    def test_get_policy_path_raises_exc(self):
        enforcer = policy.Enforcer(self.conf, policy_file='raise_error.json')
        e = self.assertRaises(cfg.ConfigFilesNotFoundError, enforcer._get_policy_path, enforcer.policy_file)
        self.assertEqual(('raise_error.json',), e.config_files)

    def test_enforcer_set_rules(self):
        self.enforcer.load_rules()
        self.enforcer.set_rules({'test': 'test1'})
        self.enforcer.load_rules()
        self.assertEqual({'test': 'test1'}, self.enforcer.rules)

    def test_enforcer_default_rule_name(self):
        enforcer = policy.Enforcer(self.conf, default_rule='foo_rule')
        self.assertEqual('foo_rule', enforcer.rules.default_rule)
        self.conf.set_override('policy_default_rule', 'bar_rule', group='oslo_policy')
        enforcer = policy.Enforcer(self.conf, default_rule='foo_rule')
        self.assertEqual('foo_rule', enforcer.rules.default_rule)
        enforcer = policy.Enforcer(self.conf)
        self.assertEqual('bar_rule', enforcer.rules.default_rule)

    def test_enforcer_register_twice_raises(self):
        self.enforcer.register_default(policy.RuleDefault(name='owner', check_str='role:owner'))
        self.assertRaises(policy.DuplicatePolicyError, self.enforcer.register_default, policy.RuleDefault(name='owner', check_str='role:owner'))

    def test_enforcer_does_not_modify_original_registered_rule(self):
        rule_original = policy.RuleDefault(name='test', check_str='role:owner')
        self.enforcer.register_default(rule_original)
        self.enforcer.registered_rules['test']._check_str = 'role:admin'
        self.enforcer.registered_rules['test']._check = 'role:admin'
        self.assertEqual(self.enforcer.registered_rules['test'].check_str, 'role:admin')
        self.assertEqual(self.enforcer.registered_rules['test'].check, 'role:admin')
        self.assertEqual(rule_original.check_str, 'role:owner')
        self.assertEqual(rule_original.check.__str__(), 'role:owner')

    def test_non_reversible_check(self):
        self.create_config_file('policy.json', jsonutils.dumps({'shared': 'field:networks:shared=True'}))
        self.enforcer.load_rules(True)
        self.assertIsNotNone(self.enforcer.rules)
        loaded_rules = jsonutils.loads(str(self.enforcer.rules))
        self.assertNotEqual('field:networks:shared=True', loaded_rules['shared'])

    def test_authorize_opt_registered(self):
        self.enforcer.register_default(policy.RuleDefault(name='test', check_str='role:test'))
        self.assertTrue(self.enforcer.authorize('test', {}, {'roles': ['test']}))

    def test_authorize_opt_not_registered(self):
        self.assertRaises(policy.PolicyNotRegistered, self.enforcer.authorize, 'test', {}, {'roles': ['test']})

    def test_enforcer_accepts_context_objects(self):
        rule = policy.RuleDefault(name='fake_rule', check_str='role:test')
        self.enforcer.register_default(rule)
        request_context = context.RequestContext()
        target_dict = {}
        self.enforcer.enforce('fake_rule', target_dict, request_context)

    def test_enforcer_accepts_subclassed_context_objects(self):
        rule = policy.RuleDefault(name='fake_rule', check_str='role:test')
        self.enforcer.register_default(rule)

        class SpecializedContext(context.RequestContext):
            pass
        request_context = SpecializedContext()
        target_dict = {}
        self.enforcer.enforce('fake_rule', target_dict, request_context)

    def test_enforcer_rejects_non_context_objects(self):
        rule = policy.RuleDefault(name='fake_rule', check_str='role:test')
        self.enforcer.register_default(rule)

        class InvalidContext(object):
            pass
        request_context = InvalidContext()
        target_dict = {}
        self.assertRaises(policy.InvalidContextObject, self.enforcer.enforce, 'fake_rule', target_dict, request_context)

    @mock.patch.object(policy.Enforcer, '_map_context_attributes_into_creds')
    def test_enforcer_call_map_context_attributes(self, map_mock):
        map_mock.return_value = {}
        rule = policy.RuleDefault(name='fake_rule', check_str='role:test')
        self.enforcer.register_default(rule)
        request_context = context.RequestContext()
        target_dict = {}
        self.enforcer.enforce('fake_rule', target_dict, request_context)
        map_mock.assert_called_once_with(request_context)

    def test_enforcer_consolidates_context_attributes_with_creds(self):
        request_context = context.RequestContext()
        expected_creds = request_context.to_policy_values()
        creds = self.enforcer._map_context_attributes_into_creds(request_context)
        for k, v in expected_creds.items():
            self.assertEqual(expected_creds[k], creds[k])

    def test_enforcer_accepts_policy_values_from_context(self):
        rule = policy.RuleDefault(name='fake_rule', check_str='role:test')
        self.enforcer.register_default(rule)
        request_context = context.RequestContext()
        policy_values = request_context.to_policy_values()
        target_dict = {}
        self.enforcer.enforce('fake_rule', target_dict, policy_values)

    def test_enforcer_understands_system_scope(self):
        self.conf.set_override('enforce_scope', True, group='oslo_policy')
        rule = policy.RuleDefault(name='fake_rule', check_str='role:test', scope_types=['system'])
        self.enforcer.register_default(rule)
        ctx = context.RequestContext(system_scope='all')
        target_dict = {}
        self.enforcer.enforce('fake_rule', target_dict, ctx)

    def test_enforcer_understands_system_scope_creds_dict(self):
        self.conf.set_override('enforce_scope', True, group='oslo_policy')
        rule = policy.RuleDefault(name='fake_rule', check_str='role:test', scope_types=['system'])
        self.enforcer.register_default(rule)
        ctx = context.RequestContext()
        creds = ctx.to_dict()
        creds['system_scope'] = 'all'
        target_dict = {}
        self.enforcer.enforce('fake_rule', target_dict, creds)

    def test_enforcer_raises_invalid_scope_with_system_scope_type(self):
        self.conf.set_override('enforce_scope', True, group='oslo_policy')
        rule = policy.RuleDefault(name='fake_rule', check_str='role:test', scope_types=['system'])
        self.enforcer.register_default(rule)
        ctx = context.RequestContext(domain_id='fake')
        target_dict = {}
        self.assertRaises(policy.InvalidScope, self.enforcer.enforce, 'fake_rule', target_dict, ctx, do_raise=True)
        self.assertFalse(self.enforcer.enforce('fake_rule', target_dict, ctx, do_raise=False))
        ctx = context.RequestContext(project_id='fake')
        self.assertRaises(policy.InvalidScope, self.enforcer.enforce, 'fake_rule', target_dict, ctx, True)
        self.assertFalse(self.enforcer.enforce('fake_rule', target_dict, ctx, do_raise=False))

    def test_enforcer_understands_domain_scope(self):
        self.conf.set_override('enforce_scope', True, group='oslo_policy')
        rule = policy.RuleDefault(name='fake_rule', check_str='role:test', scope_types=['domain'])
        self.enforcer.register_default(rule)
        ctx = context.RequestContext(domain_id='fake')
        target_dict = {}
        self.enforcer.enforce('fake_rule', target_dict, ctx)

    def test_enforcer_raises_invalid_scope_with_domain_scope_type(self):
        self.conf.set_override('enforce_scope', True, group='oslo_policy')
        rule = policy.RuleDefault(name='fake_rule', check_str='role:test', scope_types=['domain'])
        self.enforcer.register_default(rule)
        ctx = context.RequestContext(system_scope='all')
        target_dict = {}
        self.assertRaises(policy.InvalidScope, self.enforcer.enforce, 'fake_rule', target_dict, ctx, True)
        self.assertFalse(self.enforcer.enforce('fake_rule', target_dict, ctx, do_raise=False))
        ctx = context.RequestContext(project_id='fake')
        self.assertRaises(policy.InvalidScope, self.enforcer.enforce, 'fake_rule', target_dict, ctx, True)
        self.assertFalse(self.enforcer.enforce('fake_rule', target_dict, ctx, do_raise=False))

    def test_enforcer_understands_project_scope(self):
        self.conf.set_override('enforce_scope', True, group='oslo_policy')
        rule = policy.RuleDefault(name='fake_rule', check_str='role:test', scope_types=['project'])
        self.enforcer.register_default(rule)
        ctx = context.RequestContext(project_id='fake')
        target_dict = {}
        self.enforcer.enforce('fake_rule', target_dict, ctx)

    def test_enforcer_raises_invalid_scope_with_project_scope_type(self):
        self.conf.set_override('enforce_scope', True, group='oslo_policy')
        rule = policy.RuleDefault(name='fake_rule', check_str='role:test', scope_types=['project'])
        self.enforcer.register_default(rule)
        ctx = context.RequestContext(system_scope='all')
        target_dict = {}
        self.assertRaises(policy.InvalidScope, self.enforcer.enforce, 'fake_rule', target_dict, ctx, True)
        self.assertFalse(self.enforcer.enforce('fake_rule', target_dict, ctx, do_raise=False))
        ctx = context.RequestContext(domain_id='fake')
        self.assertRaises(policy.InvalidScope, self.enforcer.enforce, 'fake_rule', target_dict, ctx, True)
        self.assertFalse(self.enforcer.enforce('fake_rule', target_dict, ctx, do_raise=False))

    def test_enforce_scope_with_subclassed_checks_when_scope_not_set(self):
        self.conf.set_override('enforce_scope', True, group='oslo_policy')
        rule = _checks.TrueCheck()
        rule.scope_types = None
        ctx = context.RequestContext(system_scope='all', roles=['admin'])
        self.enforcer.enforce(rule, {}, ctx)

    def test_enforcer_raises_invalid_scope_with_subclassed_checks(self):
        self.conf.set_override('enforce_scope', True, group='oslo_policy')
        rule = _checks.TrueCheck()
        rule.scope_types = ['domain']
        ctx = context.RequestContext(system_scope='all', roles=['admin'])
        self.assertRaises(policy.InvalidScope, self.enforcer.enforce, rule, {}, ctx, do_raise=True)
        self.assertFalse(self.enforcer.enforce(rule, {}, ctx, do_raise=False))