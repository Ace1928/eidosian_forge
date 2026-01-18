import logging
import uuid
from osc_lib import exceptions
from osc_lib import utils as oscutils
from unittest import mock
from manilaclient import api_versions
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_group_snapshot_create(self):
    arglist = [self.share_group.id]
    verifylist = [('share_group', self.share_group.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.group_snapshot_mocks.create.assert_called_with(self.share_group, name=None, description=None)
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)