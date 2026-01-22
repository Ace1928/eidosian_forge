import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_target as bm_vol_target
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestListBaremetalVolumeTarget(TestBaremetalVolumeTarget):

    def setUp(self):
        super(TestListBaremetalVolumeTarget, self).setUp()
        self.baremetal_mock.volume_target.list.return_value = [baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.VOLUME_TARGET), loaded=True)]
        self.cmd = bm_vol_target.ListBaremetalVolumeTarget(self.app, None)

    def test_baremetal_volume_target_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None}
        self.baremetal_mock.volume_target.list.assert_called_once_with(**kwargs)
        collist = ('UUID', 'Node UUID', 'Driver Volume Type', 'Boot Index', 'Volume ID')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_volume_target_uuid, baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_volume_target_volume_type, baremetal_fakes.baremetal_volume_target_boot_index, baremetal_fakes.baremetal_volume_target_volume_id),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_volume_target_list_node(self):
        arglist = ['--node', baremetal_fakes.baremetal_uuid]
        verifylist = [('node', baremetal_fakes.baremetal_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'node': baremetal_fakes.baremetal_uuid, 'marker': None, 'limit': None}
        self.baremetal_mock.volume_target.list.assert_called_once_with(**kwargs)
        collist = ('UUID', 'Node UUID', 'Driver Volume Type', 'Boot Index', 'Volume ID')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_volume_target_uuid, baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_volume_target_volume_type, baremetal_fakes.baremetal_volume_target_boot_index, baremetal_fakes.baremetal_volume_target_volume_id),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_volume_target_list_long(self):
        arglist = ['--long']
        verifylist = [('detail', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'detail': True, 'marker': None, 'limit': None}
        self.baremetal_mock.volume_target.list.assert_called_with(**kwargs)
        collist = ('UUID', 'Node UUID', 'Driver Volume Type', 'Properties', 'Boot Index', 'Extra', 'Volume ID', 'Created At', 'Updated At')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_volume_target_uuid, baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_volume_target_volume_type, baremetal_fakes.baremetal_volume_target_properties, baremetal_fakes.baremetal_volume_target_boot_index, baremetal_fakes.baremetal_volume_target_extra, baremetal_fakes.baremetal_volume_target_volume_id, '', ''),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_volume_target_list_fields(self):
        arglist = ['--fields', 'uuid', 'boot_index']
        verifylist = [('fields', [['uuid', 'boot_index']])]
        fake_vt = copy.deepcopy(baremetal_fakes.VOLUME_TARGET)
        fake_vt.pop('volume_type')
        fake_vt.pop('extra')
        fake_vt.pop('properties')
        fake_vt.pop('volume_id')
        fake_vt.pop('node_uuid')
        self.baremetal_mock.volume_target.list.return_value = [baremetal_fakes.FakeBaremetalResource(None, fake_vt, loaded=True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'detail': False, 'marker': None, 'limit': None, 'fields': ('uuid', 'boot_index')}
        self.baremetal_mock.volume_target.list.assert_called_with(**kwargs)
        collist = ('UUID', 'Boot Index')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_volume_target_uuid, baremetal_fakes.baremetal_volume_target_boot_index),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_volume_target_list_fields_multiple(self):
        arglist = ['--fields', 'uuid', 'boot_index', '--fields', 'extra']
        verifylist = [('fields', [['uuid', 'boot_index'], ['extra']])]
        fake_vt = copy.deepcopy(baremetal_fakes.VOLUME_TARGET)
        fake_vt.pop('volume_type')
        fake_vt.pop('properties')
        fake_vt.pop('volume_id')
        fake_vt.pop('node_uuid')
        self.baremetal_mock.volume_target.list.return_value = [baremetal_fakes.FakeBaremetalResource(None, fake_vt, loaded=True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'detail': False, 'marker': None, 'limit': None, 'fields': ('uuid', 'boot_index', 'extra')}
        self.baremetal_mock.volume_target.list.assert_called_with(**kwargs)
        collist = ('UUID', 'Boot Index', 'Extra')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_volume_target_uuid, baremetal_fakes.baremetal_volume_target_boot_index, baremetal_fakes.baremetal_volume_target_extra),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_volume_target_list_invalid_fields(self):
        arglist = ['--fields', 'uuid', 'invalid']
        verifylist = [('fields', [['uuid', 'invalid']])]
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_volume_target_list_marker(self):
        arglist = ['--marker', baremetal_fakes.baremetal_volume_target_uuid]
        verifylist = [('marker', baremetal_fakes.baremetal_volume_target_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': baremetal_fakes.baremetal_volume_target_uuid, 'limit': None}
        self.baremetal_mock.volume_target.list.assert_called_once_with(**kwargs)

    def test_baremetal_volume_target_list_limit(self):
        arglist = ['--limit', '10']
        verifylist = [('limit', 10)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': 10}
        self.baremetal_mock.volume_target.list.assert_called_once_with(**kwargs)

    def test_baremetal_volume_target_list_sort(self):
        arglist = ['--sort', 'boot_index']
        verifylist = [('sort', 'boot_index')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None}
        self.baremetal_mock.volume_target.list.assert_called_once_with(**kwargs)

    def test_baremetal_volume_target_list_sort_desc(self):
        arglist = ['--sort', 'boot_index:desc']
        verifylist = [('sort', 'boot_index:desc')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None}
        self.baremetal_mock.volume_target.list.assert_called_once_with(**kwargs)

    def test_baremetal_volume_target_list_exclusive_options(self):
        arglist = ['--fields', 'uuid', '--long']
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, [])

    def test_baremetal_volume_target_list_negative_limit(self):
        arglist = ['--limit', '-1']
        verifylist = [('limit', -1)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)