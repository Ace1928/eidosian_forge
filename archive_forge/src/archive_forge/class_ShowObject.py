import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
class ShowObject(command.ShowOne):
    _description = _('Display object details')

    def get_parser(self, prog_name):
        parser = super(ShowObject, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', help=_('Display <object> from <container>'))
        parser.add_argument('object', metavar='<object>', help=_('Object to display'))
        return parser

    def take_action(self, parsed_args):
        data = self.app.client_manager.object_store.object_show(container=parsed_args.container, object=parsed_args.object)
        if 'properties' in data:
            data['properties'] = format_columns.DictColumn(data['properties'])
        return zip(*sorted(data.items()))