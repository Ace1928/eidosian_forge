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
def test_snapshot_adopt_name(self):
    name = 'name-' + uuid.uuid4().hex
    arglist = [self.share.id, self.export_location.fake_path, '--name', name]
    verifylist = [('share', self.share.id), ('provider_location', self.export_location.fake_path), ('name', name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.snapshots_mock.manage.assert_called_with(share=self.share, provider_location=self.export_location.fake_path, driver_options={}, name=name, description=None)
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)