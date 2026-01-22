import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
from manilaclient.common import constants
from manilaclient.osc import utils
class ListShareBackup(command.Lister):
    """List share backups."""
    _description = _('List share backups')

    def get_parser(self, prog_name):
        parser = super(ListShareBackup, self).get_parser(prog_name)
        parser.add_argument('--share', metavar='<share>', default=None, help=_('Name or ID of the share to list backups for.'))
        parser.add_argument('--name', metavar='<name>', default=None, help=_('Filter results by name. Default=None.'))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('Filter results by description. Default=None.'))
        parser.add_argument('--name~', metavar='<name~>', default=None, help=_('Filter results matching a share backup name pattern. '))
        parser.add_argument('--description~', metavar='<description~>', default=None, help=_('Filter results matching a share backup description '))
        parser.add_argument('--status', metavar='<status>', default=None, help=_('Filter results by status. Default=None.'))
        parser.add_argument('--limit', metavar='<num-backups>', type=int, default=None, action=parseractions.NonNegativeAction, help=_('Limit the number of backups returned. Default=None.'))
        parser.add_argument('--offset', metavar='<offset>', default=None, help='Start position of backup records listing.')
        parser.add_argument('--sort-key', '--sort_key', metavar='<sort_key>', type=str, default=None, help='Key to be sorted, available keys are %(keys)s. Default=None.' % {'keys': constants.BACKUP_SORT_KEY_VALUES})
        parser.add_argument('--sort-dir', '--sort_dir', metavar='<sort_dir>', type=str, default=None, help='Sort direction, available values are %(values)s. OPTIONAL: Default=None.' % {'values': constants.SORT_DIR_VALUES})
        parser.add_argument('--detail', dest='detail', metavar='<0|1>', nargs='?', type=int, const=1, default=0, help='Show detailed information about share backups.')
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_id = None
        if parsed_args.share:
            share_id = osc_utils.find_resource(share_client.shares, parsed_args.share).id
        columns = ['ID', 'Name', 'Share ID', 'Status']
        if parsed_args.detail:
            columns.extend(['Description', 'Size', 'Created At', 'Updated At', 'Availability Zone', 'Progress', 'Restore Progress', 'Host', 'Topic'])
        search_opts = {'limit': parsed_args.limit, 'offset': parsed_args.offset, 'name': parsed_args.name, 'description': parsed_args.description, 'status': parsed_args.status, 'share_id': share_id}
        search_opts['name~'] = getattr(parsed_args, 'name~')
        search_opts['description~'] = getattr(parsed_args, 'description~')
        backups = share_client.share_backups.list(detailed=parsed_args.detail, search_opts=search_opts, sort_key=parsed_args.sort_key, sort_dir=parsed_args.sort_dir)
        return (columns, (osc_utils.get_item_properties(b, columns) for b in backups))