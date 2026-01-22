import copy
from unittest import mock
from unittest.mock import call
from magnumclient.osc.v1 import nodegroups as osc_nodegroups
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
class TestNodeGroupShow(TestNodeGroup):

    def setUp(self):
        super(TestNodeGroupShow, self).setUp()
        self.nodegroup = magnum_fakes.FakeNodeGroup.create_one_nodegroup()
        self.ng_mock.get = mock.Mock()
        self.ng_mock.get.return_value = self.nodegroup
        self.data = tuple(map(lambda x: getattr(self.nodegroup, x), osc_nodegroups.NODEGROUP_ATTRIBUTES))
        self.cmd = osc_nodegroups.ShowNodeGroup(self.app, None)

    def test_nodegroup_show_pass(self):
        arglist = ['fake-cluster', 'fake-nodegroup']
        verifylist = [('cluster', 'fake-cluster'), ('nodegroup', 'fake-nodegroup')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.ng_mock.get.assert_called_with('fake-cluster', 'fake-nodegroup')
        self.assertEqual(osc_nodegroups.NODEGROUP_ATTRIBUTES, columns)
        self.assertEqual(self.data, data)

    def test_nodegroup_show_no_nodegroup_fail(self):
        arglist = ['fake-cluster']
        verifylist = [('cluster', 'fake-cluster'), ('nodegroup', '')]
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)

    def test_nodegroup_show_no_args(self):
        arglist = []
        verifylist = [('cluster', ''), ('nodegroup', '')]
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)