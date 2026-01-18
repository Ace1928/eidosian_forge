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
def test_share_group_create_share_types(self):
    share_types = manila_fakes.FakeShareType.create_share_types(count=2)
    self.share_types_mock.get = manila_fakes.FakeShareType.get_share_types(share_types)
    arglist = ['--share-types', share_types[0].id, share_types[1].id]
    verifylist = [('share_types', [share_types[0].id, share_types[1].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.groups_mock.create.assert_called_with(name=None, description=None, share_types=share_types, share_group_type=None, share_network=None, source_share_group_snapshot=None, availability_zone=None)
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)