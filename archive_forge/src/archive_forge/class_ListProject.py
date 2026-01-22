import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListProject(command.Lister):
    _description = _('List projects')

    def get_parser(self, prog_name):
        parser = super(ListProject, self).get_parser(prog_name)
        parser.add_argument('--long', action='store_true', default=False, help=_('List additional fields in output'))
        parser.add_argument('--sort', metavar='<key>[:<direction>]', help=_('Sort output by selected keys and directions (asc or desc) (default: asc), repeat this option to specify multiple keys and directions.'))
        return parser

    def take_action(self, parsed_args):
        if parsed_args.long:
            columns = ('ID', 'Name', 'Description', 'Enabled')
        else:
            columns = ('ID', 'Name')
        data = self.app.client_manager.identity.tenants.list()
        if parsed_args.sort:
            data = utils.sort_items(data, parsed_args.sort)
        return (columns, (utils.get_item_properties(s, columns, formatters={}) for s in data))