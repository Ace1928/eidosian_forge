import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListMetadefObjects(command.Lister):
    _description = _('List metadef objects inside a specific namespace.')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('namespace', metavar='<namespace>', help=_('Namespace (name) for the namespace'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        namespace = parsed_args.namespace
        columns = ['name', 'description']
        md_objects = list(image_client.metadef_objects(namespace))
        column_headers = columns
        return (column_headers, (utils.get_item_properties(md_object, columns) for md_object in md_objects))