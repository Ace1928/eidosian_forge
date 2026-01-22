import argparse
import logging
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListConsistencyGroup(command.Lister):
    _description = _('List consistency groups.')

    def get_parser(self, prog_name):
        parser = super(ListConsistencyGroup, self).get_parser(prog_name)
        parser.add_argument('--all-projects', action='store_true', help=_('Show details for all projects. Admin only. (defaults to False)'))
        parser.add_argument('--long', action='store_true', help=_('List additional fields in output'))
        return parser

    def take_action(self, parsed_args):
        if parsed_args.long:
            columns = ['ID', 'Status', 'Availability Zone', 'Name', 'Description', 'Volume Types']
        else:
            columns = ['ID', 'Status', 'Name']
        volume_client = self.app.client_manager.volume
        consistency_groups = volume_client.consistencygroups.list(detailed=True, search_opts={'all_tenants': parsed_args.all_projects})
        return (columns, (utils.get_item_properties(s, columns, formatters={'Volume Types': format_columns.ListColumn}) for s in consistency_groups))