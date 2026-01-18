from unittest import mock
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_access_rules as osc_share_access_rules
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_access_delete_unrestrict_not_available(self):
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.79')
    arglist = [self.share.id, self.access_rule.id, '--unrestrict']
    verifylist = [('share', self.share.id), ('id', self.access_rule.id), ('unrestrict', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)