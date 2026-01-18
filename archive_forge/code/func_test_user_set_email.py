from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import user
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_user_set_email(self):
    arglist = ['--email', 'barney@example.com', self.fake_user.name]
    verifylist = [('name', None), ('password', None), ('email', 'barney@example.com'), ('project', None), ('enable', False), ('disable', False), ('user', self.fake_user.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'email': 'barney@example.com', 'enabled': True}
    self.users_mock.update.assert_called_with(self.fake_user.id, **kwargs)
    self.assertIsNone(result)