from unittest import mock
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_access_rules as osc_share_access_rules
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_access_create_properties(self):
    arglist = [self.share.id, 'user', 'demo', '--properties', 'key=value']
    verifylist = [('share', self.share.id), ('access_type', 'user'), ('access_to', 'demo'), ('properties', ['key=value'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.shares_mock.get.assert_called_with(self.share.id)
    self.share.allow.assert_called_with(access_type='user', access='demo', access_level=None, metadata={'key': 'value'})
    self.assertEqual(ACCESS_RULE_ATTRIBUTES, columns)
    self.assertCountEqual(self.access_rule._info.values(), data)