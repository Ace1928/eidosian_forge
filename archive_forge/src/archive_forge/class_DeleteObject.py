import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
class DeleteObject(command.Command):
    _description = _('Delete object from container')

    def get_parser(self, prog_name):
        parser = super(DeleteObject, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', help=_('Delete object(s) from <container>'))
        parser.add_argument('objects', metavar='<object>', nargs='+', help=_('Object(s) to delete'))
        return parser

    def take_action(self, parsed_args):
        for obj in parsed_args.objects:
            self.app.client_manager.object_store.object_delete(container=parsed_args.container, object=obj)