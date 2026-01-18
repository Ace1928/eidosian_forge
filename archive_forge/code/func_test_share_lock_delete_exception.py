from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import resource_locks as osc_resource_locks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_lock_delete_exception(self):
    arglist = [self.lock.id]
    verifylist = [('lock', [self.lock.id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.lock.delete.side_effect = exceptions.CommandError()
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)