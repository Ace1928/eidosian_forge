from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_replicas as osc_share_replicas
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_replica_create_scheduler_hint_invalid_hint(self):
    arglist = [self.share.id, '--availability-zone', self.share.availability_zone, '--scheduler-hint', 'fake_hint=host1@backend1#pool1']
    verifylist = [('share', self.share.id), ('availability_zone', self.share.availability_zone), ('scheduler_hint', {'fake_hint': 'host1@backend1#pool1'})]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)