import logging
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowMetadefNamespace(command.ShowOne):
    _description = _('Show a metadef namespace')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('namespace', metavar='<namespace>', help=_('Metadef namespace to show (name)'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        namespace = parsed_args.namespace
        data = image_client.get_metadef_namespace(namespace)
        info = _format_namespace(data)
        return zip(*sorted(info.items()))