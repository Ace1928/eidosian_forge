import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
from manilaclient.common import constants
from manilaclient.osc import utils
class UnsetShareBackup(command.Command):
    """Unset share backup properties."""
    _description = _('Unset share backup properties')

    def get_parser(self, prog_name):
        parser = super(UnsetShareBackup, self).get_parser(prog_name)
        parser.add_argument('backup', metavar='<backup>', help=_('Name or ID of the backup to unset a property for'))
        parser.add_argument('--name', action='store_true', help=_('Unset a name to the backup.'))
        parser.add_argument('--description', action='store_true', help=_('Unset a description to the backup.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_backup = osc_utils.find_resource(share_client.share_backups, parsed_args.backup)
        kwargs = {}
        if parsed_args.name:
            kwargs['name'] = None
        if parsed_args.description:
            kwargs['description'] = None
        if not kwargs:
            msg = 'Either name or description must be provided.'
            raise exceptions.CommandError(msg)
        try:
            share_client.share_backups.update(share_backup, **kwargs)
        except Exception as e:
            LOG.error(_("Failed to unset share backup properties '%(properties)s': %(exception)s"), {'properties': kwargs, 'exception': e})