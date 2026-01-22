import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
from manilaclient.common import constants
from manilaclient.osc import utils
class SetShareBackup(command.Command):
    """Set share backup properties."""
    _description = _('Set share backup properties')

    def get_parser(self, prog_name):
        parser = super(SetShareBackup, self).get_parser(prog_name)
        parser.add_argument('backup', metavar='<backup>', help=_('Name or ID of the backup to set a property for'))
        parser.add_argument('--name', metavar='<name>', default=None, help=_('Set a name to the backup.'))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('Set a description to the backup.'))
        parser.add_argument('--status', metavar='<status>', choices=['available', 'error', 'creating', 'deleting', 'restoring'], help=_('Assign a status to the backup(Admin only). Options include : available, error, creating, deleting, restoring.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        result = 0
        share_backup = osc_utils.find_resource(share_client.share_backups, parsed_args.backup)
        kwargs = {}
        if parsed_args.name is not None:
            kwargs['name'] = parsed_args.name
        if parsed_args.description is not None:
            kwargs['description'] = parsed_args.description
        try:
            share_client.share_backups.update(share_backup, **kwargs)
        except Exception as e:
            result += 1
            LOG.error(_("Failed to set share backup properties '%(properties)s': %(exception)s"), {'properties': kwargs, 'exception': e})
        if parsed_args.status:
            try:
                share_client.share_backups.reset_status(share_backup, parsed_args.status)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to update backup status to '%(status)s': %(e)s"), {'status': parsed_args.status, 'e': e})
        if result > 0:
            raise exceptions.CommandError(_('One or more of the set operations failed'))