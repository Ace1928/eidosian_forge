import copy
import json
import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class RemovePortFromRouter(command.Command):
    _description = _('Remove a port from a router')

    def get_parser(self, prog_name):
        parser = super(RemovePortFromRouter, self).get_parser(prog_name)
        parser.add_argument('router', metavar='<router>', help=_('Router from which port will be removed (name or ID)'))
        parser.add_argument('port', metavar='<port>', help=_('Port to be removed and deleted (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        port = client.find_port(parsed_args.port, ignore_missing=False)
        client.remove_interface_from_router(client.find_router(parsed_args.router, ignore_missing=False), port_id=port.id)