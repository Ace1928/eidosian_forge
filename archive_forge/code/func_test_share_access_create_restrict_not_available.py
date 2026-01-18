from unittest import mock
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_access_rules as osc_share_access_rules
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.data({'lock_visibility': True, 'lock_deletion': False}, {'lock_visibility': False, 'lock_deletion': True})
@ddt.unpack
def test_share_access_create_restrict_not_available(self, lock_visibility, lock_deletion):
    arglist = [self.share.id, 'user', 'demo']
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.79')
    verifylist = [('share', self.share.id), ('access_type', 'user'), ('access_to', 'demo'), ('lock_visibility', lock_visibility), ('lock_deletion', lock_deletion), ('lock_reason', None)]
    if lock_visibility:
        arglist.append('--lock-visibility')
    if lock_deletion:
        arglist.append('--lock-deletion')
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)