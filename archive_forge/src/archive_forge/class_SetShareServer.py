import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import cliutils
from manilaclient.common import constants
class SetShareServer(command.Command):
    """Set share server properties."""
    _description = _('Set share server properties (Admin only).')

    def get_parser(self, prog_name):
        parser = super(SetShareServer, self).get_parser(prog_name)
        allowed_update_choices = ['unmanage_starting', 'server_migrating_to', 'error', 'unmanage_error', 'manage_error', 'inactive', 'active', 'server_migrating', 'manage_starting', 'deleting', 'network_change']
        allowed_update_choices_str = ', '.join(allowed_update_choices)
        parser.add_argument('share_server', metavar='<share-server>', help=_('ID of the share server to modify.'))
        parser.add_argument('--status', metavar='<status>', required=False, default=constants.STATUS_ACTIVE, help=_('Assign a status to the share server. Options include: %s. If no state is provided, active will be used.' % allowed_update_choices_str))
        parser.add_argument('--task-state', metavar='<task-state>', required=False, default=None, help=_('Indicate which task state to assign the share server. Options include migration_starting, migration_in_progress, migration_completing, migration_success, migration_error, migration_cancelled, migration_driver_in_progress, migration_driver_phase1_done, data_copying_starting, data_copying_in_progress, data_copying_completing, data_copying_completed, data_copying_cancelled, data_copying_error. '))
        return parser

    def take_action(self, parsed_args):
        if not parsed_args.status and (not parsed_args.task_state):
            msg = _('A status or a task state should be provided for this command.')
            LOG.error(msg)
            raise exceptions.CommandError(msg)
        share_client = self.app.client_manager.share
        share_server = osc_utils.find_resource(share_client.share_servers, parsed_args.share_server)
        if parsed_args.status:
            try:
                share_client.share_servers.reset_state(share_server, parsed_args.status)
            except Exception as e:
                msg = (_("Failed to set status '%(status)s': %(exception)s"), {'status': parsed_args.status, 'exception': e})
                LOG.error(msg)
                raise exceptions.CommandError(msg)
        if parsed_args.task_state:
            if share_client.api_version < api_versions.APIVersion('2.57'):
                raise exceptions.CommandError('Setting the state of a share server is only available with manila API version >= 2.57')
            else:
                result = 0
                try:
                    share_client.share_servers.reset_task_state(share_server, parsed_args.task_state)
                except Exception as e:
                    LOG.error(_('Failed to update share server task state %s'), e)
                    result += 1
            if result > 0:
                raise exceptions.CommandError(_('One or more of the reset operations failed'))