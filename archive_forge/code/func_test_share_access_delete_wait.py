from unittest import mock
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_access_rules as osc_share_access_rules
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_access_delete_wait(self):
    arglist = [self.share.id, self.access_rule.id, '--wait']
    verifylist = [('share', self.share.id), ('id', self.access_rule.id), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch('osc_lib.utils.wait_for_delete', return_value=True):
        result = self.cmd.take_action(parsed_args)
        self.shares_mock.get.assert_called_with(self.share.id)
        self.share.deny.assert_called_with(self.access_rule.id)
        self.assertIsNone(result)