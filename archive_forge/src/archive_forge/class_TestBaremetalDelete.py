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
class TestBaremetalDelete(TestBaremetal):

    def setUp(self):
        super(TestBaremetalDelete, self).setUp()
        self.baremetal_mock.node.get.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.BAREMETAL), loaded=True)
        self.cmd = baremetal_node.DeleteBaremetalNode(self.app, None)

    def test_baremetal_delete(self):
        arglist = ['xxx-xxxxxx-xxxx']
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = 'xxx-xxxxxx-xxxx'
        self.baremetal_mock.node.delete.assert_called_with(args)

    def test_baremetal_delete_multiple(self):
        arglist = ['xxx-xxxxxx-xxxx', 'fakename']
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = ['xxx-xxxxxx-xxxx', 'fakename']
        self.baremetal_mock.node.delete.assert_has_calls([mock.call(x) for x in args])
        self.assertEqual(2, self.baremetal_mock.node.delete.call_count)

    def test_baremetal_delete_multiple_with_failure(self):
        arglist = ['xxx-xxxxxx-xxxx', 'badname']
        verifylist = []
        self.baremetal_mock.node.delete.side_effect = ['', exc.ClientException]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exc.ClientException, self.cmd.take_action, parsed_args)
        args = ['xxx-xxxxxx-xxxx', 'badname']
        self.baremetal_mock.node.delete.assert_has_calls([mock.call(x) for x in args])
        self.assertEqual(2, self.baremetal_mock.node.delete.call_count)