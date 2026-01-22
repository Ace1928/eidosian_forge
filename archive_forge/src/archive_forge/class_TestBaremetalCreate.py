import copy
import io
import json
import sys
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient.common import utils as commonutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_node
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
from ironicclient.v1 import utils as v1_utils
class TestBaremetalCreate(TestBaremetal):

    def setUp(self):
        super(TestBaremetalCreate, self).setUp()
        self.baremetal_mock.node.create.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.BAREMETAL), loaded=True)
        self.cmd = baremetal_node.CreateBaremetalNode(self.app, None)
        self.arglist = ['--driver', 'fake_driver']
        self.verifylist = [('driver', 'fake_driver')]
        self.collist = ('chassis_uuid', 'instance_uuid', 'maintenance', 'name', 'power_state', 'provision_state', 'uuid')
        self.datalist = (baremetal_fakes.baremetal_chassis_uuid_empty, baremetal_fakes.baremetal_instance_uuid, baremetal_fakes.baremetal_maintenance, baremetal_fakes.baremetal_name, baremetal_fakes.baremetal_power_state, baremetal_fakes.baremetal_provision_state, baremetal_fakes.baremetal_uuid)
        self.actual_kwargs = {'driver': 'fake_driver'}

    def check_with_options(self, addl_arglist, addl_verifylist, addl_kwargs):
        arglist = copy.copy(self.arglist) + addl_arglist
        verifylist = copy.copy(self.verifylist) + addl_verifylist
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        collist = copy.copy(self.collist)
        self.assertEqual(collist, columns)
        datalist = copy.copy(self.datalist)
        self.assertEqual(datalist, tuple(data))
        kwargs = copy.copy(self.actual_kwargs)
        kwargs.update(addl_kwargs)
        self.baremetal_mock.node.create.assert_called_once_with(**kwargs)

    def test_baremetal_create_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_create_with_driver(self):
        arglist = copy.copy(self.arglist)
        verifylist = copy.copy(self.verifylist)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        collist = copy.copy(self.collist)
        self.assertEqual(collist, columns)
        self.assertNotIn('ports', columns)
        self.assertNotIn('states', columns)
        datalist = copy.copy(self.datalist)
        self.assertEqual(datalist, tuple(data))
        kwargs = copy.copy(self.actual_kwargs)
        self.baremetal_mock.node.create.assert_called_once_with(**kwargs)

    def test_baremetal_create_with_chassis(self):
        self.check_with_options(['--chassis-uuid', 'chassis_uuid'], [('chassis_uuid', 'chassis_uuid')], {'chassis_uuid': 'chassis_uuid'})

    def test_baremetal_create_with_driver_info(self):
        self.check_with_options(['--driver-info', 'arg1=val1', '--driver-info', 'arg2=val2'], [('driver_info', ['arg1=val1', 'arg2=val2'])], {'driver_info': {'arg1': 'val1', 'arg2': 'val2'}})

    def test_baremetal_create_with_properties(self):
        self.check_with_options(['--property', 'arg1=val1', '--property', 'arg2=val2'], [('properties', ['arg1=val1', 'arg2=val2'])], {'properties': {'arg1': 'val1', 'arg2': 'val2'}})

    def test_baremetal_create_with_extra(self):
        self.check_with_options(['--extra', 'arg1=val1', '--extra', 'arg2=val2'], [('extra', ['arg1=val1', 'arg2=val2'])], {'extra': {'arg1': 'val1', 'arg2': 'val2'}})

    def test_baremetal_create_with_uuid(self):
        self.check_with_options(['--uuid', 'uuid'], [('uuid', 'uuid')], {'uuid': 'uuid'})

    def test_baremetal_create_with_name(self):
        self.check_with_options(['--name', 'name'], [('name', 'name')], {'name': 'name'})

    def test_baremetal_create_with_bios_interface(self):
        self.check_with_options(['--bios-interface', 'bios'], [('bios_interface', 'bios')], {'bios_interface': 'bios'})

    def test_baremetal_create_with_boot_interface(self):
        self.check_with_options(['--boot-interface', 'boot'], [('boot_interface', 'boot')], {'boot_interface': 'boot'})

    def test_baremetal_create_with_console_interface(self):
        self.check_with_options(['--console-interface', 'console'], [('console_interface', 'console')], {'console_interface': 'console'})

    def test_baremetal_create_with_deploy_interface(self):
        self.check_with_options(['--deploy-interface', 'deploy'], [('deploy_interface', 'deploy')], {'deploy_interface': 'deploy'})

    def test_baremetal_create_with_firmware_interface(self):
        self.check_with_options(['--firmware-interface', 'firmware'], [('firmware_interface', 'firmware')], {'firmware_interface': 'firmware'})

    def test_baremetal_create_with_inspect_interface(self):
        self.check_with_options(['--inspect-interface', 'inspect'], [('inspect_interface', 'inspect')], {'inspect_interface': 'inspect'})

    def test_baremetal_create_with_management_interface(self):
        self.check_with_options(['--management-interface', 'management'], [('management_interface', 'management')], {'management_interface': 'management'})

    def test_baremetal_create_with_network_data(self):
        self.check_with_options(['--network-data', '{"a": ["b"]}'], [('network_data', '{"a": ["b"]}')], {'network_data': {'a': ['b']}})

    def test_baremetal_create_with_network_interface(self):
        self.check_with_options(['--network-interface', 'neutron'], [('network_interface', 'neutron')], {'network_interface': 'neutron'})

    def test_baremetal_create_with_power_interface(self):
        self.check_with_options(['--power-interface', 'power'], [('power_interface', 'power')], {'power_interface': 'power'})

    def test_baremetal_create_with_raid_interface(self):
        self.check_with_options(['--raid-interface', 'raid'], [('raid_interface', 'raid')], {'raid_interface': 'raid'})

    def test_baremetal_create_with_rescue_interface(self):
        self.check_with_options(['--rescue-interface', 'rescue'], [('rescue_interface', 'rescue')], {'rescue_interface': 'rescue'})

    def test_baremetal_create_with_storage_interface(self):
        self.check_with_options(['--storage-interface', 'storage'], [('storage_interface', 'storage')], {'storage_interface': 'storage'})

    def test_baremetal_create_with_vendor_interface(self):
        self.check_with_options(['--vendor-interface', 'vendor'], [('vendor_interface', 'vendor')], {'vendor_interface': 'vendor'})

    def test_baremetal_create_with_resource_class(self):
        self.check_with_options(['--resource-class', 'foo'], [('resource_class', 'foo')], {'resource_class': 'foo'})

    def test_baremetal_create_with_conductor_group(self):
        self.check_with_options(['--conductor-group', 'conductor_group'], [('conductor_group', 'conductor_group')], {'conductor_group': 'conductor_group'})

    def test_baremetal_create_with_automated_clean(self):
        self.check_with_options(['--automated-clean'], [('automated_clean', True)], {'automated_clean': True})

    def test_baremetal_create_with_no_automated_clean(self):
        self.check_with_options(['--no-automated-clean'], [('automated_clean', False)], {'automated_clean': False})

    def test_baremetal_create_with_owner(self):
        self.check_with_options(['--owner', 'owner 1'], [('owner', 'owner 1')], {'owner': 'owner 1'})

    def test_baremetal_create_with_description(self):
        self.check_with_options(['--description', 'there is no spoon'], [('description', 'there is no spoon')], {'description': 'there is no spoon'})

    def test_baremetal_create_with_lessee(self):
        self.check_with_options(['--lessee', 'lessee 1'], [('lessee', 'lessee 1')], {'lessee': 'lessee 1'})

    def test_baremetal_create_with_shard(self):
        self.check_with_options(['--shard', 'myshard'], [('shard', 'myshard')], {'shard': 'myshard'})

    def test_baremetal_create_with_parent_node(self):
        self.check_with_options(['--parent-node', 'nodex'], [('parent_node', 'nodex')], {'parent_node': 'nodex'})