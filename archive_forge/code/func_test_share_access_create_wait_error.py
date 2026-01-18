from unittest import mock
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_access_rules as osc_share_access_rules
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@mock.patch('manilaclient.osc.v2.share_access_rules.LOG')
def test_share_access_create_wait_error(self, mock_logger):
    arglist = [self.share.id, 'user', 'demo', '--wait']
    verifylist = [('share', self.share.id), ('access_type', 'user'), ('access_to', 'demo'), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch('osc_lib.utils.wait_for_status', return_value=False):
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.get.assert_called_with(self.share.id)
        self.share.allow.assert_called_with(access_type='user', access='demo', access_level=None, metadata={})
        mock_logger.error.assert_called_with('ERROR: Share access rule is in error state.')
        self.assertEqual(ACCESS_RULE_ATTRIBUTES, columns)
        self.assertCountEqual(self.access_rule._info.values(), data)