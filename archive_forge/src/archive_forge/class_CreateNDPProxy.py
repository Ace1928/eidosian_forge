import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class CreateNDPProxy(command.ShowOne):
    _description = _('Create NDP proxy')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('router', metavar='<router>', help=_('The name or ID of a router'))
        parser.add_argument('--name', metavar='<name>', help=_('New NDP proxy name'))
        parser.add_argument('--port', metavar='<port>', required=True, help=_('The name or ID of the network port associated to the NDP proxy'))
        parser.add_argument('--ip-address', metavar='<ip-address>', help=_('The IPv6 address that is to be proxied. In case the port has multiple addresses assigned, use this option to select which address is to be used.'))
        parser.add_argument('--description', metavar='<description>', help=_('A text to describe/contextualize the use of the NDP proxy configuration'))
        return parser

    def take_action(self, parsed_args):
        attrs = {'name': parsed_args.name}
        client = self.app.client_manager.network
        router = client.find_router(parsed_args.router, ignore_missing=False)
        attrs['router_id'] = router.id
        if parsed_args.ip_address:
            attrs['ip_address'] = parsed_args.ip_address
        port = client.find_port(parsed_args.port, ignore_missing=False)
        attrs['port_id'] = port.id
        if parsed_args.description is not None:
            attrs['description'] = parsed_args.description
        obj = client.create_ndp_proxy(**attrs)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns)
        return (display_columns, data)