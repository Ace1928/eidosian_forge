from openstackclient.tests.functional import base
class CommandTest(base.TestCase):
    """Functional tests for openstackclient command list."""
    GROUPS = ['openstack.volume.v3', 'openstack.network.v2', 'openstack.image.v2', 'openstack.identity.v3', 'openstack.compute.v2', 'openstack.common', 'openstack.cli']

    def test_command_list_no_option(self):
        cmd_output = self.openstack('command list', parse_output=True)
        group_names = [each.get('Command Group') for each in cmd_output]
        for one_group in self.GROUPS:
            self.assertIn(one_group, group_names)

    def test_command_list_with_group(self):
        input_groups = ['volume', 'network', 'image', 'identity', 'compute.v2']
        for each_input in input_groups:
            cmd_output = self.openstack('command list --group %s' % each_input, parse_output=True)
            group_names = [each.get('Command Group') for each in cmd_output]
            for each_name in group_names:
                self.assertIn(each_input, each_name)