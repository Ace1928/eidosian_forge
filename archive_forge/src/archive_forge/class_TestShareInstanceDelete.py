from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_instances as osc_share_instances
from manilaclient import api_versions
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareInstanceDelete(TestShareInstance):

    def setUp(self):
        super(TestShareInstanceDelete, self).setUp()
        self.share_instance = manila_fakes.FakeShareInstance.create_one_share_instance()
        self.instances_mock.get.return_value = self.share_instance
        self.cmd = osc_share_instances.ShareInstanceDelete(self.app, None)

    def test_share_instance_delete_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_instance_delete(self):
        arglist = [self.share_instance.id]
        verifylist = [('instance', [self.share_instance.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.instances_mock.force_delete.assert_called_with(self.share_instance)
        self.assertIsNone(result)

    def test_share_instance_delete_multiple(self):
        share_instances = manila_fakes.FakeShareInstance.create_share_instances(count=2)
        instance_ids = [instance.id for instance in share_instances]
        arglist = instance_ids
        verifylist = [('instance', instance_ids)]
        self.instances_mock.get.side_effect = share_instances
        delete_calls = [mock.call(instance) for instance in share_instances]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.instances_mock.force_delete.assert_has_calls(delete_calls)
        self.assertEqual(self.instances_mock.force_delete.call_count, len(share_instances))
        self.assertIsNone(result)

    def test_share_instance_delete_exception(self):
        arglist = [self.share_instance.id]
        verifylist = [('instance', [self.share_instance.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.instances_mock.force_delete.side_effect = exceptions.CommandError()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_instance_delete_wait(self):
        arglist = [self.share_instance.id, '--wait']
        verifylist = [('instance', [self.share_instance.id]), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_delete', return_value=True):
            result = self.cmd.take_action(parsed_args)
            self.instances_mock.force_delete.assert_called_with(self.share_instance)
            self.instances_mock.get.assert_called_with(self.share_instance.id)
            self.assertIsNone(result)

    def test_share_instance_delete_wait_exception(self):
        arglist = [self.share_instance.id, '--wait']
        verifylist = [('instance', [self.share_instance.id]), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_delete', return_value=False):
            self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)