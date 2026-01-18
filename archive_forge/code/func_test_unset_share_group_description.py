import argparse
from unittest import mock
import uuid
from osc_lib import exceptions
from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as oscutils
from manilaclient.osc import utils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_groups as osc_share_groups
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_unset_share_group_description(self):
    arglist = [self.share_group.id, '--description']
    verifylist = [('share_group', self.share_group.id), ('description', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.groups_mock.update.assert_called_with(self.share_group, description=None)
    self.assertIsNone(result)