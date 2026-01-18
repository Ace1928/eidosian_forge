from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.common.apiclient.exceptions import NotFound
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_types as osc_share_types
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_type_delete_multiple(self):
    arglist = []
    for t in self.share_types:
        arglist.append(t.id)
    verifylist = [('share_types', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = []
    for t in self.share_types:
        calls.append(mock.call(t))
    self.shares_mock.delete.assert_has_calls(calls)
    self.assertIsNone(result)