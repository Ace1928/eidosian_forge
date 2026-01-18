import contextlib
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import user
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_user_list_domain(self):
    arglist = ['--domain', self.domain.id]
    verifylist = [('domain', self.domain.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'domain': self.domain.id, 'group': None}
    self.users_mock.list.assert_called_with(**kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, tuple(data))