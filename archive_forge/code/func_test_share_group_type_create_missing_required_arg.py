from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.common.apiclient.exceptions import NotFound
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_group_types as osc_share_group_types
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_group_type_create_missing_required_arg(self):
    """Verifies missing required arguments."""
    arglist = [self.share_group_type.name]
    verifylist = [('name', self.share_group_type.name)]
    self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)