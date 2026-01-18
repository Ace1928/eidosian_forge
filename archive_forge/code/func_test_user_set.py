from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from openstackclient.tests.functional.identity.v2 import common
def test_user_set(self):
    username = self._create_dummy_user()
    raw_output = self.openstack('user show %s' % username)
    user = self.parse_show_as_object(raw_output)
    new_username = data_utils.rand_name('NewTestUser')
    new_email = data_utils.rand_name() + '@example.com'
    raw_output = self.openstack('user set --email %(email)s --name %(new_name)s %(id)s' % {'email': new_email, 'new_name': new_username, 'id': user['id']})
    self.assertEqual(0, len(raw_output))
    raw_output = self.openstack('user show %s' % new_username)
    new_user = self.parse_show_as_object(raw_output)
    self.assertEqual(user['id'], new_user['id'])
    self.assertEqual(new_email, new_user['email'])