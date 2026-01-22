import logging
from osc_lib.command import command
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class SetFloatingIPCommand(command.ShowOne):
    """Set floatingip ptr record"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('floatingip_id', help='Floating IP ID in format region:floatingip_id')
        parser.add_argument('ptrdname', help='PTRD Name')
        description_group = parser.add_mutually_exclusive_group()
        description_group.add_argument('--description', help='Description')
        description_group.add_argument('--no-description', action='store_true')
        ttl_group = parser.add_mutually_exclusive_group()
        ttl_group.add_argument('--ttl', type=int, help='TTL')
        ttl_group.add_argument('--no-ttl', action='store_true')
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        data = {}
        if parsed_args.no_description:
            data['description'] = None
        elif parsed_args.description:
            data['description'] = parsed_args.description
        if parsed_args.no_ttl:
            data['ttl'] = None
        elif parsed_args.ttl:
            data['ttl'] = parsed_args.ttl
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        fip = client.floatingips.set(parsed_args.floatingip_id, parsed_args.ptrdname, parsed_args.description, parsed_args.ttl)
        _format_floatingip(fip)
        return zip(*sorted(fip.items()))