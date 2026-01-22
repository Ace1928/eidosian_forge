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
class TestBootdeviceShow(TestBaremetal):

    def setUp(self):
        super(TestBootdeviceShow, self).setUp()
        self.cmd = baremetal_node.BootdeviceShowBaremetalNode(self.app, None)
        self.baremetal_mock.node.get_boot_device.return_value = {'boot_device': 'pxe', 'persistent': False}
        self.baremetal_mock.node.get_supported_boot_devices.return_value = {'supported_boot_devices': v1_utils.BOOT_DEVICES}

    def test_bootdevice_show(self):
        arglist = ['node_uuid']
        verifylist = [('node', 'node_uuid')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.get_boot_device.assert_called_once_with('node_uuid')

    def test_bootdevice_supported_show(self):
        arglist = ['node_uuid', '--supported']
        verifylist = [('node', 'node_uuid'), ('supported', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        mock = self.baremetal_mock.node.get_supported_boot_devices
        mock.assert_called_once_with('node_uuid')