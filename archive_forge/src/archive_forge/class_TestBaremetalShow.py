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
class TestBaremetalShow(TestBaremetal):

    def setUp(self):
        super(TestBaremetalShow, self).setUp()
        self.baremetal_mock.node.get.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.BAREMETAL), loaded=True)
        self.baremetal_mock.node.get_by_instance_uuid.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.BAREMETAL), loaded=True)
        self.cmd = baremetal_node.ShowBaremetalNode(self.app, None)

    def test_baremetal_show(self):
        arglist = ['xxx-xxxxxx-xxxx']
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        args = ['xxx-xxxxxx-xxxx']
        self.baremetal_mock.node.get.assert_called_with(*args, fields=None)
        collist = ('chassis_uuid', 'instance_uuid', 'maintenance', 'name', 'power_state', 'provision_state', 'uuid')
        self.assertEqual(collist, columns)
        self.assertNotIn('ports', columns)
        self.assertNotIn('states', columns)
        datalist = (baremetal_fakes.baremetal_chassis_uuid_empty, baremetal_fakes.baremetal_instance_uuid, baremetal_fakes.baremetal_maintenance, baremetal_fakes.baremetal_name, baremetal_fakes.baremetal_power_state, baremetal_fakes.baremetal_provision_state, baremetal_fakes.baremetal_uuid)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_show_no_node(self):
        arglist = []
        verifylist = []
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_show_with_instance_uuid(self):
        arglist = ['xxx-xxxxxx-xxxx', '--instance']
        verifylist = [('instance_uuid', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = ['xxx-xxxxxx-xxxx']
        self.baremetal_mock.node.get_by_instance_uuid.assert_called_with(*args, fields=None)

    def test_baremetal_show_fields(self):
        arglist = ['xxxxx', '--fields', 'uuid', 'name']
        verifylist = [('node', 'xxxxx'), ('fields', [['uuid', 'name']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertNotIn('chassis_uuid', columns)
        args = ['xxxxx']
        fields = ['uuid', 'name']
        self.baremetal_mock.node.get.assert_called_with(*args, fields=fields)

    def test_baremetal_show_fields_multiple(self):
        arglist = ['xxxxx', '--fields', 'uuid', 'name', '--fields', 'extra']
        verifylist = [('node', 'xxxxx'), ('fields', [['uuid', 'name'], ['extra']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertNotIn('chassis_uuid', columns)
        args = ['xxxxx']
        fields = ['uuid', 'name', 'extra']
        self.baremetal_mock.node.get.assert_called_with(*args, fields=fields)

    def test_baremetal_show_invalid_fields(self):
        arglist = ['xxxxx', '--fields', 'uuid', 'invalid']
        verifylist = [('node', 'xxxxx'), ('fields', [['uuid', 'invalid']])]
        self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)