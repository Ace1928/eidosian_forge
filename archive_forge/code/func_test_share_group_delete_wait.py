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
def test_share_group_delete_wait(self):
    arglist = [self.share_group.id, '--wait']
    verifylist = [('share_group', [self.share_group.id]), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch('osc_lib.utils.wait_for_delete', return_value=True):
        result = self.cmd.take_action(parsed_args)
        self.groups_mock.delete.assert_called_with(self.share_group, force=False)
        self.groups_mock.get.assert_called_with(self.share_group.id)
        self.assertIsNone(result)