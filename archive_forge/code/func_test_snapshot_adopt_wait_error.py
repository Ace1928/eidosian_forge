from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import share_snapshots as osc_share_snapshots
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_snapshot_adopt_wait_error(self):
    arglist = [self.share.id, self.export_location.fake_path, '--wait']
    verifylist = [('share', self.share.id), ('provider_location', self.export_location.fake_path), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch('osc_lib.utils.wait_for_status', return_value=False):
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.get.assert_called_with(self.share_snapshot.id)
        self.snapshots_mock.manage.assert_called_with(share=self.share, provider_location=self.export_location.fake_path, driver_options={}, name=None, description=None)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)