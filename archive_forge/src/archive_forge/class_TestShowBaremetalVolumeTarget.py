import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_target as bm_vol_target
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestShowBaremetalVolumeTarget(TestBaremetalVolumeTarget):

    def setUp(self):
        super(TestShowBaremetalVolumeTarget, self).setUp()
        self.baremetal_mock.volume_target.get.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.VOLUME_TARGET), loaded=True)
        self.cmd = bm_vol_target.ShowBaremetalVolumeTarget(self.app, None)

    def test_baremetal_volume_target_show(self):
        arglist = ['vvv-tttttt-vvvv']
        verifylist = [('volume_target', baremetal_fakes.baremetal_volume_target_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        args = ['vvv-tttttt-vvvv']
        self.baremetal_mock.volume_target.get.assert_called_once_with(*args, fields=None)
        collist = ('boot_index', 'extra', 'node_uuid', 'properties', 'uuid', 'volume_id', 'volume_type')
        self.assertEqual(collist, columns)
        datalist = (baremetal_fakes.baremetal_volume_target_boot_index, baremetal_fakes.baremetal_volume_target_extra, baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_volume_target_properties, baremetal_fakes.baremetal_volume_target_uuid, baremetal_fakes.baremetal_volume_target_volume_id, baremetal_fakes.baremetal_volume_target_volume_type)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_volume_target_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_volume_target_show_fields(self):
        arglist = ['vvv-tttttt-vvvv', '--fields', 'uuid', 'volume_id']
        verifylist = [('fields', [['uuid', 'volume_id']]), ('volume_target', baremetal_fakes.baremetal_volume_target_uuid)]
        fake_vt = copy.deepcopy(baremetal_fakes.VOLUME_TARGET)
        fake_vt.pop('node_uuid')
        fake_vt.pop('volume_type')
        fake_vt.pop('boot_index')
        fake_vt.pop('extra')
        fake_vt.pop('properties')
        self.baremetal_mock.volume_target.get.return_value = baremetal_fakes.FakeBaremetalResource(None, fake_vt, loaded=True)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        args = ['vvv-tttttt-vvvv']
        fields = ['uuid', 'volume_id']
        self.baremetal_mock.volume_target.get.assert_called_once_with(*args, fields=fields)
        collist = ('uuid', 'volume_id')
        self.assertEqual(collist, columns)
        datalist = (baremetal_fakes.baremetal_volume_target_uuid, baremetal_fakes.baremetal_volume_target_volume_id)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_volume_target_show_fields_multiple(self):
        arglist = ['vvv-tttttt-vvvv', '--fields', 'uuid', 'volume_id', '--fields', 'volume_type']
        verifylist = [('fields', [['uuid', 'volume_id'], ['volume_type']]), ('volume_target', baremetal_fakes.baremetal_volume_target_uuid)]
        fake_vt = copy.deepcopy(baremetal_fakes.VOLUME_TARGET)
        fake_vt.pop('node_uuid')
        fake_vt.pop('boot_index')
        fake_vt.pop('extra')
        fake_vt.pop('properties')
        self.baremetal_mock.volume_target.get.return_value = baremetal_fakes.FakeBaremetalResource(None, fake_vt, loaded=True)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        args = ['vvv-tttttt-vvvv']
        fields = ['uuid', 'volume_id', 'volume_type']
        self.baremetal_mock.volume_target.get.assert_called_once_with(*args, fields=fields)
        collist = ('uuid', 'volume_id', 'volume_type')
        self.assertEqual(collist, columns)
        datalist = (baremetal_fakes.baremetal_volume_target_uuid, baremetal_fakes.baremetal_volume_target_volume_id, baremetal_fakes.baremetal_volume_target_volume_type)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_volume_target_show_invalid_fields(self):
        arglist = ['vvv-tttttt-vvvv', '--fields', 'uuid', 'invalid']
        verifylist = None
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)