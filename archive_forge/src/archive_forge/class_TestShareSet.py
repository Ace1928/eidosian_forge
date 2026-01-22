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
class TestShareSet(TestShare):

    def setUp(self):
        super(TestShareSet, self).setUp()
        self._share = manila_fakes.FakeShare.create_one_share(methods={'reset_state': None, 'reset_task_state': None, 'set_metadata': None})
        self.shares_mock.get.return_value = self._share
        self.cmd = osc_shares.SetShare(self.app, None)

    def test_share_set_property(self):
        arglist = ['--property', 'Zorilla=manila', self._share.id]
        verifylist = [('property', {'Zorilla': 'manila'}), ('share', self._share.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self._share.set_metadata.assert_called_with({'Zorilla': 'manila'})

    def test_share_set_name(self):
        new_name = uuid.uuid4().hex
        arglist = ['--name', new_name, self._share.id]
        verifylist = [('name', new_name), ('share', self._share.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.shares_mock.update.assert_called_with(self._share.id, display_name=parsed_args.name)

    def test_share_set_description(self):
        new_description = uuid.uuid4().hex
        arglist = ['--description', new_description, self._share.id]
        verifylist = [('description', new_description), ('share', self._share.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.shares_mock.update.assert_called_with(self._share.id, display_description=parsed_args.description)

    def test_share_set_visibility(self):
        arglist = ['--public', 'true', self._share.id]
        verifylist = [('public', 'true'), ('share', self._share.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.shares_mock.update.assert_called_with(self._share.id, is_public='true')

    def test_share_set_visibility_exception(self):
        arglist = ['--public', 'not_a_boolean_value', self._share.id]
        verifylist = [('public', 'not_a_boolean_value'), ('share', self._share.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.shares_mock.update.assert_called_with(self._share.id, is_public='not_a_boolean_value')
        self.shares_mock.update.side_effect = exceptions.BadRequest()
        self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_set_property_exception(self):
        arglist = ['--property', 'key=', self._share.id]
        verifylist = [('property', {'key': ''}), ('share', self._share.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self._share.set_metadata.assert_called_with({'key': ''})
        self._share.set_metadata.side_effect = exceptions.BadRequest
        self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_set_status(self):
        new_status = 'available'
        arglist = [self._share.id, '--status', new_status]
        verifylist = [('share', self._share.id), ('status', new_status)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self._share.reset_state.assert_called_with(new_status)
        self.assertIsNone(result)

    def test_share_set_status_exception(self):
        new_status = 'available'
        arglist = [self._share.id, '--status', new_status]
        verifylist = [('share', self._share.id), ('status', new_status)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self._share.reset_state.side_effect = Exception()
        self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_set_task_state(self):
        new_task_state = 'migration_starting'
        arglist = [self._share.id, '--task-state', new_task_state]
        verifylist = [('share', self._share.id), ('task_state', new_task_state)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self._share.reset_task_state.assert_called_with(new_task_state)
        self.assertIsNone(result)