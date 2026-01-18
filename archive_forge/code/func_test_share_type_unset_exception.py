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
def test_share_type_unset_exception(self):
    arglist = [self.share_type.id, 'snapshot_support']
    verifylist = [('share_type', self.share_type.id), ('extra_specs', ['snapshot_support'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.share_type.unset_keys.side_effect = NotFound()
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)