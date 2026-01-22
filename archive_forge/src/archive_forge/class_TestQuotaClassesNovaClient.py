from tempest.lib import exceptions
from novaclient.tests.functional import base
class TestQuotaClassesNovaClient(base.ClientTestBase):
    """Nova quota classes functional tests for the v2.1 microversion."""
    COMPUTE_API_VERSION = '2.1'
    _included_resources = ['instances', 'cores', 'ram', 'floating_ips', 'fixed_ips', 'metadata_items', 'injected_files', 'injected_file_content_bytes', 'injected_file_path_bytes', 'key_pairs', 'security_groups', 'security_group_rules']
    _excluded_resources = ['server_groups', 'server_group_members']
    _extra_update_resources = _excluded_resources
    _blocked_update_resources = []

    def _get_quota_class_name(self):
        """Returns a fake quota class name specific to this test class."""
        return 'fake-class-%s' % self.COMPUTE_API_VERSION.replace('.', '-')

    def _verify_quota_class_show_output(self, output, expected_values):
        for quota_name in self._included_resources:
            self.assertIn(quota_name, expected_values)
            expected_value = expected_values[quota_name]
            actual_value = self._get_value_from_the_table(output, quota_name)
            self.assertEqual(expected_value, actual_value)
        for quota_name in self._excluded_resources:
            self.assertRaises(ValueError, self._get_value_from_the_table, output, quota_name)

    def test_quota_class_show(self):
        """Tests showing quota class values for a fake non-existing quota
        class. The API will return the defaults if the quota class does not
        actually exist. We use a fake class to avoid any interaction with the
        real default quota class values.
        """
        default_quota_class_set = self.client.quota_classes.get('default')
        default_values = {quota_name: str(getattr(default_quota_class_set, quota_name)) for quota_name in self._included_resources}
        output = self.nova('quota-class-show %s' % self._get_quota_class_name())
        self._verify_quota_class_show_output(output, default_values)

    def test_quota_class_update(self):
        """Tests updating a fake quota class. The way this works in the API
        is that if the quota class is not found, it is created. So in this
        test we can use a fake quota class with fake values and they will all
        get set. We don't use the default quota class because it is global
        and we don't want to interfere with other tests.
        """
        class_name = self._get_quota_class_name()
        params = [class_name]
        expected_values = {}
        for quota_name in self._included_resources + self._extra_update_resources:
            params.append('--%s 99' % quota_name.replace('_', '-'))
            expected_values[quota_name] = '99'
        self.nova('quota-class-update', params=' '.join(params))
        output = self.nova('quota-class-show %s' % class_name)
        self._verify_quota_class_show_output(output, expected_values)
        for quota_name in self._blocked_update_resources:
            self.assertRaises(exceptions.CommandFailed, self.nova, 'quota-class-update %s --%s 99' % (class_name, quota_name.replace('_', '-')))