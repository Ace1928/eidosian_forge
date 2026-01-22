import argparse
import getpass
import io
import json
import logging
import os
from cliff import columns as cliff_columns
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common as network_common
class RemovePort(command.Command):
    _description = _('Remove port from server')

    def get_parser(self, prog_name):
        parser = super(RemovePort, self).get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', help=_('Server to remove the port from (name or ID)'))
        parser.add_argument('port', metavar='<port>', help=_('Port to remove from the server (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        server = compute_client.find_server(parsed_args.server, ignore_missing=False)
        if self.app.client_manager.is_network_endpoint_enabled():
            network_client = self.app.client_manager.network
            port_id = network_client.find_port(parsed_args.port, ignore_missing=False).id
        else:
            port_id = parsed_args.port
        compute_client.delete_server_interface(port_id, server=server, ignore_missing=False)