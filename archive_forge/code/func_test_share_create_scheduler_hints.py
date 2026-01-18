import argparse
import ddt
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.api_versions import MAX_VERSION
from manilaclient.common.apiclient import exceptions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import share as osc_shares
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_create_scheduler_hints(self):
    """Verifies scheduler hints are parsed correctly."""
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.65')
    shares = self.setup_shares_mock(count=2)
    share1_name = shares[0].name
    share2_name = shares[1].name
    arglist = [self.new_share.share_proto, str(self.new_share.size), '--share-type', self.share_type.id, '--scheduler-hint', 'same_host=%s' % share1_name, '--scheduler-hint', 'different_host=%s' % share2_name]
    verifylist = [('share_proto', self.new_share.share_proto), ('size', self.new_share.size), ('share_type', self.share_type.id), ('scheduler_hint', {'same_host': share1_name, 'different_host': share2_name})]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.shares_mock.create.assert_called_with(availability_zone=None, description=None, is_public=False, metadata={}, name=None, share_group_id=None, share_network=None, share_proto=self.new_share.share_proto, share_type=self.share_type.id, size=self.new_share.size, snapshot_id=None, scheduler_hints={'same_host': shares[0].id, 'different_host': shares[1].id})
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.datalist, data)