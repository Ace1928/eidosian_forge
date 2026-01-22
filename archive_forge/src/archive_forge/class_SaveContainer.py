import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
class SaveContainer(command.Command):
    _description = _('Save container contents locally')

    def get_parser(self, prog_name):
        parser = super(SaveContainer, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', help=_('Container to save'))
        return parser

    def take_action(self, parsed_args):
        self.app.client_manager.object_store.container_save(container=parsed_args.container)