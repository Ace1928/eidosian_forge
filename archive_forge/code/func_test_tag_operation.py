import uuid
from openstackclient.tests.functional import base
def test_tag_operation(self):
    cmd_output = self.openstack('token issue ', parse_output=True)
    auth_project_id = cmd_output['project_id']
    name1 = self._create_resource_and_tag_check('', [])
    name2 = self._create_resource_and_tag_check('--tag red --tag blue', ['red', 'blue'])
    name3 = self._create_resource_and_tag_check('--no-tag', [])
    self._set_resource_and_tag_check('set', name1, '--tag red --tag green', ['red', 'green'])
    list_expected = ((name1, ['red', 'green']), (name2, ['red', 'blue']), (name3, []))
    self._list_tag_check(auth_project_id, list_expected)
    self._set_resource_and_tag_check('set', name1, '--tag blue', ['red', 'green', 'blue'])
    self._set_resource_and_tag_check('set', name1, '--no-tag --tag yellow --tag orange --tag purple', ['yellow', 'orange', 'purple'])
    self._set_resource_and_tag_check('unset', name1, '--tag yellow', ['orange', 'purple'])
    self._set_resource_and_tag_check('unset', name1, '--all-tag', [])
    self._set_resource_and_tag_check('set', name2, '--no-tag', [])