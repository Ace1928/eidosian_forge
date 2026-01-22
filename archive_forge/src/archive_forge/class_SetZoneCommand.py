import logging
from osc_lib.command import command
from osc_lib import exceptions as osc_exc
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class SetZoneCommand(command.ShowOne):
    """Set zone properties"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('id', help='Zone ID')
        parser.add_argument('--email', help='Zone Email')
        parser.add_argument('--ttl', type=int, help='Time To Live (Seconds)')
        description_group = parser.add_mutually_exclusive_group()
        description_group.add_argument('--description', help='Description')
        description_group.add_argument('--no-description', action='store_true')
        parser.add_argument('--masters', help='Zone Masters', nargs='+')
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        data = {}
        if parsed_args.email:
            data['email'] = parsed_args.email
        if parsed_args.ttl:
            data['ttl'] = parsed_args.ttl
        if parsed_args.no_description:
            data['description'] = None
        elif parsed_args.description:
            data['description'] = parsed_args.description
        if parsed_args.masters:
            data['masters'] = parsed_args.masters
        updated = client.zones.update(parsed_args.id, data)
        _format_zone(updated)
        return zip(*sorted(updated.items()))