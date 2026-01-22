import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_connector as bm_vol_connector
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestUnsetBaremetalVolumeConnector(TestBaremetalVolumeConnector):

    def setUp(self):
        super(TestUnsetBaremetalVolumeConnector, self).setUp()
        self.cmd = bm_vol_connector.UnsetBaremetalVolumeConnector(self.app, None)

    def test_baremetal_volume_connector_unset_extra(self):
        arglist = [baremetal_fakes.baremetal_volume_connector_uuid, '--extra', 'key1']
        verifylist = [('volume_connector', baremetal_fakes.baremetal_volume_connector_uuid), ('extra', ['key1'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_connector.update.assert_called_once_with(baremetal_fakes.baremetal_volume_connector_uuid, [{'path': '/extra/key1', 'op': 'remove'}])

    def test_baremetal_volume_connector_unset_multiple_extras(self):
        arglist = [baremetal_fakes.baremetal_volume_connector_uuid, '--extra', 'key1', '--extra', 'key2']
        verifylist = [('volume_connector', baremetal_fakes.baremetal_volume_connector_uuid), ('extra', ['key1', 'key2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_connector.update.assert_called_once_with(baremetal_fakes.baremetal_volume_connector_uuid, [{'path': '/extra/key1', 'op': 'remove'}, {'path': '/extra/key2', 'op': 'remove'}])

    def test_baremetal_volume_connector_unset_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_volume_connector_unset_no_property(self):
        arglist = [baremetal_fakes.baremetal_volume_connector_uuid]
        verifylist = [('volume_connector', baremetal_fakes.baremetal_volume_connector_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_connector.update.assert_not_called()