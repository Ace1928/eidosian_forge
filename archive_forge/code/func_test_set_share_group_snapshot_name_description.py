import logging
import uuid
from osc_lib import exceptions
from osc_lib import utils as oscutils
from unittest import mock
from manilaclient import api_versions
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_set_share_group_snapshot_name_description(self):
    group_snapshot_name = 'group-snapshot-name-' + uuid.uuid4().hex
    group_snapshot_description = 'group-snapshot-description-' + uuid.uuid4().hex
    arglist = [self.share_group_snapshot.id, '--name', group_snapshot_name, '--description', group_snapshot_description]
    verifylist = [('share_group_snapshot', self.share_group_snapshot.id), ('name', group_snapshot_name), ('description', group_snapshot_description)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.group_snapshot_mocks.update.assert_called_with(self.share_group_snapshot, name=parsed_args.name, description=parsed_args.description)
    self.assertIsNone(result)