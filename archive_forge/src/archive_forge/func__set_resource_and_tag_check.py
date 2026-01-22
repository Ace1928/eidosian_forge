import uuid
from openstackclient.tests.functional import base
def _set_resource_and_tag_check(self, command, name, args, expected):
    cmd_output = self.openstack('{} {} {} {}'.format(self.base_command, command, args, name))
    self.assertFalse(cmd_output)
    cmd_output = self.openstack('{} show {}'.format(self.base_command, name), parse_output=True)
    self.assertEqual(set(expected), set(cmd_output['tags']))