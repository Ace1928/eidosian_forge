from novaclient.tests.functional import base
class TestServersResize(base.ClientTestBase):
    """Servers resize functional tests."""
    COMPUTE_API_VERSION = '2.1'

    def _compare_quota_usage(self, old_usage, new_usage, expect_diff=True):
        """Compares the quota usage in the provided AbsoluteLimits."""
        self.assertEqual(old_usage['totalInstancesUsed'], new_usage['totalInstancesUsed'], 'totalInstancesUsed does not match')
        self.assertEqual(old_usage['totalCoresUsed'], new_usage['totalCoresUsed'], 'totalCoresUsed does not match')
        if expect_diff:
            self.assertNotEqual(old_usage['totalRAMUsed'], new_usage['totalRAMUsed'], 'totalRAMUsed should have changed')
        else:
            self.assertEqual(old_usage['totalRAMUsed'], new_usage['totalRAMUsed'], 'totalRAMUsed does not match')

    def test_resize_up_confirm(self):
        """Tests creating a server and resizes up and confirms the resize.
        Compares quota before, during and after the resize.
        """
        server_id = self._create_server(flavor=self.flavor.id).id
        starting_usage = self._get_absolute_limits()
        alternate_flavor = self._pick_alternate_flavor()
        self.nova('resize', params='%s %s --poll' % (server_id, alternate_flavor))
        resize_usage = self._get_absolute_limits()
        self._compare_quota_usage(starting_usage, resize_usage)
        self.nova('resize-confirm', params='%s' % server_id)
        self._wait_for_state_change(server_id, 'active')
        confirm_usage = self._get_absolute_limits()
        self._compare_quota_usage(resize_usage, confirm_usage, expect_diff=False)

    def _create_resize_down_flavors(self):
        """Creates two flavors with different size ram but same size vcpus
        and disk.

        :returns: tuple of 2 IDs which represents larger_flavor for resize and
            smaller flavor.
        """
        output = self.nova('flavor-create', params='%s auto 128 0 1' % self.name_generate())
        larger_id = self._get_column_value_from_single_row_table(output, 'ID')
        self.addCleanup(self.nova, 'flavor-delete', params=larger_id)
        output = self.nova('flavor-create', params='%s auto 64 0 1' % self.name_generate())
        smaller_id = self._get_column_value_from_single_row_table(output, 'ID')
        self.addCleanup(self.nova, 'flavor-delete', params=smaller_id)
        return (larger_id, smaller_id)

    def test_resize_down_revert(self):
        """Tests creating a server and resizes down and reverts the resize.
        Compares quota before, during and after the resize.
        """
        larger_flavor, smaller_flavor = self._create_resize_down_flavors()
        server_id = self._create_server(flavor=larger_flavor).id
        starting_usage = self._get_absolute_limits()
        self.nova('resize', params='%s %s --poll' % (server_id, smaller_flavor))
        resize_usage = self._get_absolute_limits()
        self._compare_quota_usage(starting_usage, resize_usage)
        self.nova('resize-revert', params='%s' % server_id)
        self._wait_for_state_change(server_id, 'active')
        revert_usage = self._get_absolute_limits()
        self._compare_quota_usage(resize_usage, revert_usage)