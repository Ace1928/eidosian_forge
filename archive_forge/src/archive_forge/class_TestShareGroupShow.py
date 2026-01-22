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
class TestShareGroupShow(TestShareGroup):

    def setUp(self):
        super(TestShareGroupShow, self).setUp()
        self.share_group = manila_fakes.FakeShareGroup.create_one_share_group()
        self.formatted_result = manila_fakes.FakeShareGroup.create_one_share_group(attrs={'id': self.share_group.id, 'created_at': self.share_group.created_at, 'project_id': self.share_group.project_id, 'share_group_type_id': self.share_group.share_group_type_id, 'share_types': '\n'.join(self.share_group.share_types)})
        self.groups_mock.get.return_value = self.share_group
        self.data = tuple(self.formatted_result._info.values())
        self.columns = tuple(self.share_group._info.keys())
        self.cmd = osc_share_groups.ShowShareGroup(self.app, None)

    def test_share_group_show(self):
        arglist = [self.share_group.id]
        verifylist = [('share_group', self.share_group.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.groups_mock.get.assert_called_with(self.share_group.id)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)