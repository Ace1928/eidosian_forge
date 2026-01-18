from novaclient.tests.functional import base
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