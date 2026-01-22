import uuid
from openstackclient.tests.functional import base
def _create_resource_and_tag_check(self, args, expected):
    name = uuid.uuid4().hex
    cmd_output = self._create_resource_for_tag_test(name, args)
    self.addCleanup(self.openstack, '{} delete {}'.format(self.base_command, name))
    self.assertIsNotNone(cmd_output['id'])
    self.assertEqual(set(expected), set(cmd_output['tags']))
    return name