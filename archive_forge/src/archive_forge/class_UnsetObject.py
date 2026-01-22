import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
class UnsetObject(command.Command):
    _description = _('Unset object properties')

    def get_parser(self, prog_name):
        parser = super(UnsetObject, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', help=_('Modify <object> from <container>'))
        parser.add_argument('object', metavar='<object>', help=_('Object to modify'))
        parser.add_argument('--property', metavar='<key>', required=True, action='append', default=[], help=_('Property to remove from object (repeat option to remove multiple properties)'))
        return parser

    def take_action(self, parsed_args):
        self.app.client_manager.object_store.object_unset(parsed_args.container, parsed_args.object, properties=parsed_args.property)