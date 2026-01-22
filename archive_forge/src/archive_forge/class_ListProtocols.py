import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListProtocols(command.Lister):
    _description = _('List federation protocols')

    def get_parser(self, prog_name):
        parser = super(ListProtocols, self).get_parser(prog_name)
        parser.add_argument('--identity-provider', metavar='<identity-provider>', required=True, help=_('Identity provider to list (name or ID) (required)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        protocols = identity_client.federation.protocols.list(parsed_args.identity_provider)
        columns = ('id', 'mapping')
        response_attributes = ('id', 'mapping_id')
        items = [utils.get_item_properties(s, response_attributes) for s in protocols]
        return (columns, items)