import operator
from unittest import mock
import warnings
from oslo_config import cfg
import stevedore
import testtools
import yaml
from oslo_policy import generator
from oslo_policy import policy
from oslo_policy.tests import base
from oslo_serialization import jsonutils
class GenerateSampleYAMLTestCase(base.PolicyBaseTestCase):

    def setUp(self):
        super(GenerateSampleYAMLTestCase, self).setUp()
        self.enforcer = policy.Enforcer(self.conf, policy_file='policy.yaml')

    def test_generate_loadable_yaml(self):
        extensions = []
        for name, opts in OPTS.items():
            ext = stevedore.extension.Extension(name=name, entry_point=None, plugin=None, obj=opts)
            extensions.append(ext)
        test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=extensions, namespace=['base_rules', 'rules'])
        output_file = self.get_config_file_fullname('policy.yaml')
        with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr) as mock_ext_mgr:
            generator._generate_sample(['base_rules', 'rules'], output_file, include_help=False)
            mock_ext_mgr.assert_called_once_with('oslo.policy.policies', names=['base_rules', 'rules'], on_load_failure_callback=generator.on_load_failure_callback, invoke_on_load=True)
        self.enforcer.load_rules()
        self.assertIn('owner', self.enforcer.rules)
        self.assertIn('admin', self.enforcer.rules)
        self.assertIn('admin_or_owner', self.enforcer.rules)
        self.assertEqual('project_id:%(project_id)s', str(self.enforcer.rules['owner']))
        self.assertEqual('is_admin:True', str(self.enforcer.rules['admin']))
        self.assertEqual('(rule:admin or rule:owner)', str(self.enforcer.rules['admin_or_owner']))

    def test_expected_content(self):
        extensions = []
        for name, opts in OPTS.items():
            ext = stevedore.extension.Extension(name=name, entry_point=None, plugin=None, obj=opts)
            extensions.append(ext)
        test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=extensions, namespace=['base_rules', 'rules'])
        expected = '# Basic admin check\n#"admin": "is_admin:True"\n\n# This is a long description to check that line wrapping functions\n# properly\n# GET  /foo/\n# POST  /test/\n#"owner": "project_id:%(project_id)s"\n\n#"shared": "field:networks:shared=True"\n\n#"admin_or_owner": "rule:admin or rule:owner"\n\n'
        output_file = self.get_config_file_fullname('policy.yaml')
        with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr) as mock_ext_mgr:
            generator._generate_sample(['base_rules', 'rules'], output_file)
            mock_ext_mgr.assert_called_once_with('oslo.policy.policies', names=['base_rules', 'rules'], on_load_failure_callback=generator.on_load_failure_callback, invoke_on_load=True)
        with open(output_file, 'r') as written_file:
            written_policy = written_file.read()
        self.assertEqual(expected, written_policy)

    def test_expected_content_stdout(self):
        extensions = []
        for name, opts in OPTS.items():
            ext = stevedore.extension.Extension(name=name, entry_point=None, plugin=None, obj=opts)
            extensions.append(ext)
        test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=extensions, namespace=['base_rules', 'rules'])
        expected = '# Basic admin check\n#"admin": "is_admin:True"\n\n# This is a long description to check that line wrapping functions\n# properly\n# GET  /foo/\n# POST  /test/\n#"owner": "project_id:%(project_id)s"\n\n#"shared": "field:networks:shared=True"\n\n#"admin_or_owner": "rule:admin or rule:owner"\n\n'
        stdout = self._capture_stdout()
        with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr) as mock_ext_mgr:
            generator._generate_sample(['base_rules', 'rules'], output_file=None)
            mock_ext_mgr.assert_called_once_with('oslo.policy.policies', names=['base_rules', 'rules'], on_load_failure_callback=generator.on_load_failure_callback, invoke_on_load=True)
        self.assertEqual(expected, stdout.getvalue())

    def test_policies_deprecated_for_removal(self):
        rule = policy.RuleDefault(name='foo:post_bar', check_str='role:fizz', description='Create a bar.', deprecated_for_removal=True, deprecated_reason='This policy is not used anymore', deprecated_since='N')
        opts = {'rules': [rule]}
        extensions = []
        for name, opts in opts.items():
            ext = stevedore.extension.Extension(name=name, entry_point=None, plugin=None, obj=opts)
            extensions.append(ext)
        test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=extensions, namespace=['rules'])
        expected = '# DEPRECATED\n# "foo:post_bar" has been deprecated since N.\n# This policy is not used anymore\n# Create a bar.\n#"foo:post_bar": "role:fizz"\n\n'
        stdout = self._capture_stdout()
        with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr) as mock_ext_mgr:
            generator._generate_sample(['rules'], output_file=None)
            mock_ext_mgr.assert_called_once_with('oslo.policy.policies', names=['rules'], on_load_failure_callback=generator.on_load_failure_callback, invoke_on_load=True)
        self.assertEqual(expected, stdout.getvalue())

    def test_deprecated_policies_are_aliased_to_new_names(self):
        deprecated_rule = policy.DeprecatedRule(name='foo:post_bar', check_str='role:fizz', deprecated_reason='foo:post_bar is being removed in favor of foo:create_bar', deprecated_since='N')
        new_rule = policy.RuleDefault(name='foo:create_bar', check_str='role:fizz', description='Create a bar.', deprecated_rule=deprecated_rule)
        opts = {'rules': [new_rule]}
        extensions = []
        for name, opts in opts.items():
            ext = stevedore.extension.Extension(name=name, entry_point=None, plugin=None, obj=opts)
            extensions.append(ext)
        test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=extensions, namespace=['rules'])
        expected = '# Create a bar.\n#"foo:create_bar": "role:fizz"\n\n# DEPRECATED\n# "foo:post_bar":"role:fizz" has been deprecated since N in favor of\n# "foo:create_bar":"role:fizz".\n# foo:post_bar is being removed in favor of foo:create_bar\n# WARNING: A rule name change has been identified.\n#          This may be an artifact of new rules being\n#          included which require legacy fallback\n#          rules to ensure proper policy behavior.\n#          Alternatively, this may just be an alias.\n#          Please evaluate on a case by case basis\n#          keeping in mind the format for aliased\n#          rules is:\n#          "old_rule_name": "new_rule_name".\n# "foo:post_bar": "rule:foo:create_bar"\n\n'
        stdout = self._capture_stdout()
        with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr) as mock_ext_mgr:
            generator._generate_sample(['rules'], output_file=None)
            mock_ext_mgr.assert_called_once_with('oslo.policy.policies', names=['rules'], on_load_failure_callback=generator.on_load_failure_callback, invoke_on_load=True)
        self.assertEqual(expected, stdout.getvalue())

    def test_deprecated_policies_with_same_name(self):
        deprecated_rule = policy.DeprecatedRule(name='foo:create_bar', check_str='role:old', deprecated_reason='role:fizz is a more sane default for foo:create_bar', deprecated_since='N')
        new_rule = policy.RuleDefault(name='foo:create_bar', check_str='role:fizz', description='Create a bar.', deprecated_rule=deprecated_rule)
        opts = {'rules': [new_rule]}
        extensions = []
        for name, opts in opts.items():
            ext = stevedore.extension.Extension(name=name, entry_point=None, plugin=None, obj=opts)
            extensions.append(ext)
        test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=extensions, namespace=['rules'])
        expected = '# Create a bar.\n#"foo:create_bar": "role:fizz"\n\n# DEPRECATED\n# "foo:create_bar":"role:old" has been deprecated since N in favor of\n# "foo:create_bar":"role:fizz".\n# role:fizz is a more sane default for foo:create_bar\n\n'
        stdout = self._capture_stdout()
        with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr) as mock_ext_mgr:
            generator._generate_sample(['rules'], output_file=None)
            mock_ext_mgr.assert_called_once_with('oslo.policy.policies', names=['rules'], on_load_failure_callback=generator.on_load_failure_callback, invoke_on_load=True)
        self.assertEqual(expected, stdout.getvalue())

    def _test_formatting(self, description, expected):
        rule = [policy.RuleDefault('admin', 'is_admin:True', description=description)]
        ext = stevedore.extension.Extension(name='check_rule', entry_point=None, plugin=None, obj=rule)
        test_mgr = stevedore.named.NamedExtensionManager.make_test_instance(extensions=[ext], namespace=['check_rule'])
        output_file = self.get_config_file_fullname('policy.yaml')
        with mock.patch('stevedore.named.NamedExtensionManager', return_value=test_mgr) as mock_ext_mgr:
            generator._generate_sample(['check_rule'], output_file)
            mock_ext_mgr.assert_called_once_with('oslo.policy.policies', names=['check_rule'], on_load_failure_callback=generator.on_load_failure_callback, invoke_on_load=True)
        with open(output_file, 'r') as written_file:
            written_policy = written_file.read()
        self.assertEqual(expected, written_policy)

    def test_empty_line_formatting(self):
        description = 'Check Summary \n\nThis is a description to check that empty line has no white spaces.'
        expected = '# Check Summary\n#\n# This is a description to check that empty line has no white spaces.\n#"admin": "is_admin:True"\n\n'
        self._test_formatting(description, expected)

    def test_paragraph_formatting(self):
        description = "\nHere's a neat description with a paragraph. We want to make sure that it wraps\nproperly.\n"
        expected = '# Here\'s a neat description with a paragraph. We want to make sure\n# that it wraps properly.\n#"admin": "is_admin:True"\n\n'
        self._test_formatting(description, expected)

    def test_literal_block_formatting(self):
        description = "Here's another description.\n\n    This one has a literal block.\n    These lines should be kept apart.\n    They should not be wrapped, even though they may be longer than 70 chars\n"
        expected = '# Here\'s another description.\n#\n#     This one has a literal block.\n#     These lines should be kept apart.\n#     They should not be wrapped, even though they may be longer than 70 chars\n#"admin": "is_admin:True"\n\n'
        self._test_formatting(description, expected)

    def test_invalid_formatting(self):
        description = "Here's a broken description.\n\nWe have some text...\n    Followed by a literal block without any spaces.\n    We don't support definition lists, so this is just wrong!\n"
        expected = '# Here\'s a broken description.\n#\n# We have some text...\n#\n#     Followed by a literal block without any spaces.\n#     We don\'t support definition lists, so this is just wrong!\n#"admin": "is_admin:True"\n\n'
        with warnings.catch_warnings(record=True) as warns:
            self._test_formatting(description, expected)
            self.assertEqual(1, len(warns))
            self.assertTrue(issubclass(warns[-1].category, FutureWarning))
            self.assertIn('Invalid policy description', str(warns[-1].message))