from novaclient.tests.functional import base
class TestNetworkCommandsV2_36(base.ClientTestBase):
    """Deprecated network command functional tests."""
    COMPUTE_API_VERSION = '2.36'

    def test_limits(self):
        """Tests that 2.36 won't return network-related resource limits and
        the CLI output won't show them.
        """
        output = self.nova('limits')
        self.assertRaises(ValueError, self._get_value_from_the_table, output, 'SecurityGroups')

    def test_quota_show(self):
        """Tests that 2.36 won't return network-related resource quotas and
        the CLI output won't show them.
        """
        output = self.nova('quota-show')
        self.assertRaises(ValueError, self._get_value_from_the_table, output, 'security_groups')