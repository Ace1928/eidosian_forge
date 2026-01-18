from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_instances as osc_share_instances
from manilaclient import api_versions
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
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