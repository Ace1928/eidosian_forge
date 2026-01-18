import contextlib
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import user
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_user_show_with_domain(self):
    user = identity_fakes.FakeUser.create_one_user({'name': self.user.name})
    identity_client = self.app.client_manager.identity
    arglist = ['--domain', self.user.domain_id, user.name]
    verifylist = [('domain', self.user.domain_id), ('user', user.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    user_str = common._get_token_resource(identity_client, 'user', parsed_args.user, parsed_args.domain)
    self.assertEqual(self.user.id, user_str)
    arglist = ['--domain', user.domain_id, user.name]
    verifylist = [('domain', user.domain_id), ('user', user.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    user_str = common._get_token_resource(identity_client, 'user', parsed_args.user, parsed_args.domain)
    self.assertEqual(user.name, user_str)