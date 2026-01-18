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
def test_share_adopt_share_server_api_version_exception(self):
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.48')
    arglist = ['some.host@driver#pool', 'NFS', '10.0.0.1:/example_path', '--share-server-id', self._share.share_server_id]
    verifylist = [('service_host', 'some.host@driver#pool'), ('protocol', 'NFS'), ('export_path', '10.0.0.1:/example_path'), ('share_server_id', self._share.share_server_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)