import logging
import uuid
from osc_lib import exceptions
from osc_lib import utils as oscutils
from unittest import mock
from manilaclient import api_versions
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_group_snapshot_delete_force(self):
    arglist = [self.share_group_snapshot.id, '--force']
    verifylist = [('share_group_snapshot', [self.share_group_snapshot.id]), ('force', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.group_snapshot_mocks.delete.assert_called_with(self.share_group_snapshot, force=True)
    self.assertIsNone(result)