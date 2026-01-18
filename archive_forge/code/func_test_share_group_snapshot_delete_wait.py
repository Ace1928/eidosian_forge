import logging
import uuid
from osc_lib import exceptions
from osc_lib import utils as oscutils
from unittest import mock
from manilaclient import api_versions
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_group_snapshot_delete_wait(self):
    arglist = [self.share_group_snapshot.id, '--wait']
    verifylist = [('share_group_snapshot', [self.share_group_snapshot.id]), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch('osc_lib.utils.wait_for_delete', return_value=True):
        result = self.cmd.take_action(parsed_args)
        self.group_snapshot_mocks.delete.assert_called_with(self.share_group_snapshot, force=False)
        self.group_snapshot_mocks.get.assert_called_with(self.share_group_snapshot.id)
        self.assertIsNone(result)