from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as osc_lib_utils
from manilaclient.common.apiclient import exceptions as api_exceptions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_snapshot_instance_show(self):
    arglist = [self.share_snapshot_instance.id]
    verifylist = [('snapshot_instance', self.share_snapshot_instance.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.share_snapshot_instances_mock.get.assert_called_with(self.share_snapshot_instance.id)
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)