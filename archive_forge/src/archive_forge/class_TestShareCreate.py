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
@ddt.ddt
class TestShareCreate(TestShare):

    def setUp(self):
        super(TestShareCreate, self).setUp()
        self.new_share = manila_fakes.FakeShare.create_one_share(attrs={'status': 'available'})
        self.shares_mock.create.return_value = self.new_share
        self.shares_mock.get.return_value = self.new_share
        self.share_snapshot = manila_fakes.FakeShareSnapshot.create_one_snapshot()
        self.snapshots_mock.get.return_value = self.share_snapshot
        self.share_type = manila_fakes.FakeShareType.create_one_sharetype()
        self.share_types_mock.get.return_value = self.share_type
        self.cmd = osc_shares.CreateShare(self.app, None)
        self.datalist = tuple(self.new_share._info.values())
        self.columns = tuple(self.new_share._info.keys())

    def test_share_create_required_args(self):
        """Verifies required arguments."""
        arglist = [self.new_share.share_proto, str(self.new_share.size), '--share-type', self.share_type.id]
        verifylist = [('share_proto', self.new_share.share_proto), ('size', self.new_share.size), ('share_type', self.share_type.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.create.assert_called_with(availability_zone=None, description=None, is_public=False, metadata={}, name=None, share_group_id=None, share_network=None, share_proto=self.new_share.share_proto, share_type=self.share_type.id, size=self.new_share.size, snapshot_id=None, scheduler_hints={})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, data)

    def test_share_create_missing_required_arg(self):
        """Verifies missing required arguments."""
        arglist = [self.new_share.share_proto]
        verifylist = [('share_proto', self.new_share.share_proto)]
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_share_create_metadata(self):
        arglist = [self.new_share.share_proto, str(self.new_share.size), '--share-type', self.share_type.id, '--property', 'Manila=zorilla', '--property', 'Zorilla=manila']
        verifylist = [('share_proto', self.new_share.share_proto), ('size', self.new_share.size), ('share_type', self.share_type.id), ('property', {'Manila': 'zorilla', 'Zorilla': 'manila'})]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.create.assert_called_with(availability_zone=None, description=None, is_public=False, metadata={'Manila': 'zorilla', 'Zorilla': 'manila'}, name=None, share_group_id=None, share_network=None, share_proto=self.new_share.share_proto, share_type=self.share_type.id, size=self.new_share.size, snapshot_id=None, scheduler_hints={})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, data)

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

    def test_share_create_with_snapshot(self):
        """Verifies create share from snapshot."""
        arglist = [self.new_share.share_proto, str(self.new_share.size), '--snapshot-id', self.share_snapshot.id]
        verifylist = [('share_proto', self.new_share.share_proto), ('size', self.new_share.size), ('snapshot_id', self.share_snapshot.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('manilaclient.common.apiclient.utils.find_resource', mock.Mock(return_value=self.share_snapshot)):
            columns, data = self.cmd.take_action(parsed_args)
            osc_shares.apiutils.find_resource.assert_called_once_with(mock.ANY, self.share_snapshot.id)
        self.shares_mock.create.assert_called_with(availability_zone=None, description=None, is_public=False, metadata={}, name=None, share_group_id=None, share_network=None, share_proto=self.new_share.share_proto, share_type=None, size=self.new_share.size, snapshot_id=self.share_snapshot.id, scheduler_hints={})
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, data)

    def test_share_create_wait(self):
        arglist = [self.new_share.share_proto, str(self.new_share.size), '--share-type', self.share_type.id, '--wait']
        verifylist = [('share_proto', self.new_share.share_proto), ('size', self.new_share.size), ('share_type', self.share_type.id), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.create.assert_called_with(availability_zone=None, description=None, is_public=False, metadata={}, name=None, share_group_id=None, share_network=None, share_proto=self.new_share.share_proto, share_type=self.share_type.id, size=self.new_share.size, snapshot_id=None, scheduler_hints={})
        self.shares_mock.get.assert_called_with(self.new_share.id)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, data)

    @mock.patch('manilaclient.osc.v2.share.LOG')
    def test_share_create_wait_error(self, mock_logger):
        arglist = [self.new_share.share_proto, str(self.new_share.size), '--share-type', self.share_type.id, '--wait']
        verifylist = [('share_proto', self.new_share.share_proto), ('size', self.new_share.size), ('share_type', self.share_type.id), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch('osc_lib.utils.wait_for_status', return_value=False):
            columns, data = self.cmd.take_action(parsed_args)
            self.shares_mock.create.assert_called_with(availability_zone=None, description=None, is_public=False, metadata={}, name=None, share_group_id=None, share_network=None, share_proto=self.new_share.share_proto, share_type=self.share_type.id, size=self.new_share.size, snapshot_id=None, scheduler_hints={})
            mock_logger.error.assert_called_with('ERROR: Share is in error state.')
            self.shares_mock.get.assert_called_with(self.new_share.id)
            self.assertCountEqual(self.columns, columns)
            self.assertCountEqual(self.datalist, data)

    def test_create_share_with_no_existing_share_type(self):
        arglist = [self.new_share.share_proto, str(self.new_share.size)]
        verifylist = [('share_proto', self.new_share.share_proto), ('size', self.new_share.size)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.share_types_mock.get.side_effect = osc_exceptions.CommandError()
        self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.share_types_mock.get.assert_called_once_with(share_type='default')

    @ddt.data('None', 'NONE', 'none')
    def test_create_share_with_the_name_none(self, name):
        arglist = ['--name', name, self.new_share.share_proto, str(self.new_share.size), '--share-type', self.share_type.id]
        verifylist = [('name', name), ('share_proto', self.new_share.share_proto), ('size', self.new_share.size), ('share_type', self.share_type.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)