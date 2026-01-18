from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import resource_locks as osc_resource_locks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_lock_set(self):
    arglist = [self.lock.id, '--resource-action', 'unmanage']
    verifylist = [('lock', self.lock.id), ('resource_action', 'unmanage')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertIsNone(result)
    self.locks_mock.update.assert_called_with(self.lock.id, resource_action='unmanage')