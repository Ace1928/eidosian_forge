import copy
from unittest import mock
from unittest.mock import call
from magnumclient.osc.v1 import nodegroups as osc_nodegroups
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
class TestNodeGroupUpdate(TestNodeGroup):

    def setUp(self):
        super(TestNodeGroupUpdate, self).setUp()
        self.ng_mock.update = mock.Mock()
        self.ng_mock.update.return_value = None
        self.cmd = osc_nodegroups.UpdateNodeGroup(self.app, None)

    def test_nodegroup_update_pass(self):
        arglist = ['foo', 'ng1', 'remove', 'bar']
        verifylist = [('cluster', 'foo'), ('nodegroup', 'ng1'), ('op', 'remove'), ('attributes', [['bar']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.ng_mock.update.assert_called_with('foo', 'ng1', [{'op': 'remove', 'path': '/bar'}])

    def test_nodegroup_update_bad_op(self):
        arglist = ['cluster', 'ng1', 'foo', 'bar']
        verifylist = [('cluster', 'cluster'), ('nodegroup', 'ng1'), ('op', 'foo'), ('attributes', ['bar'])]
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)