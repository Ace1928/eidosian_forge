from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_instances as osc_share_instances
from manilaclient import api_versions
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_instance_delete_wait(self):
    arglist = [self.share_instance.id, '--wait']
    verifylist = [('instance', [self.share_instance.id]), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch('osc_lib.utils.wait_for_delete', return_value=True):
        result = self.cmd.take_action(parsed_args)
        self.instances_mock.force_delete.assert_called_with(self.share_instance)
        self.instances_mock.get.assert_called_with(self.share_instance.id)
        self.assertIsNone(result)