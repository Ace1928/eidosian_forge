import logging
import uuid
from osc_lib import exceptions
from osc_lib import utils as oscutils
from unittest import mock
from manilaclient import api_versions
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_group_snapshot_delete_exception(self):
    arglist = [self.share_group_snapshot.id]
    verifylist = [('share_group_snapshot', [self.share_group_snapshot.id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.group_snapshot_mocks.delete.side_effect = exceptions.CommandError()
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)