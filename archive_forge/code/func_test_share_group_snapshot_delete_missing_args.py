import logging
import uuid
from osc_lib import exceptions
from osc_lib import utils as oscutils
from unittest import mock
from manilaclient import api_versions
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_group_snapshot_delete_missing_args(self):
    arglist = []
    verifylist = []
    self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)