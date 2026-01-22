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
class AddFixedIP(command.ShowOne):
    _description = _('Add fixed IP address to server')

    def get_parser(self, prog_name):
        parser = super(AddFixedIP, self).get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', help=_('Server to receive the fixed IP address (name or ID)'))
        parser.add_argument('network', metavar='<network>', help=_('Network to allocate the fixed IP address from (name or ID)'))
        parser.add_argument('--fixed-ip-address', metavar='<ip-address>', help=_('Requested fixed IP address'))
        parser.add_argument('--tag', metavar='<tag>', help=_('Tag for the attached interface. (supported by --os-compute-api-version 2.49 or above)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        server = compute_client.find_server(parsed_args.server, ignore_missing=False)
        if parsed_args.tag:
            if not sdk_utils.supports_microversion(compute_client, '2.49'):
                msg = _('--os-compute-api-version 2.49 or greater is required to support the --tag option')
                raise exceptions.CommandError(msg)
        if self.app.client_manager.is_network_endpoint_enabled():
            network_client = self.app.client_manager.network
            net_id = network_client.find_network(parsed_args.network, ignore_missing=False).id
        else:
            net_id = parsed_args.network
        kwargs = {'net_id': net_id}
        if parsed_args.fixed_ip_address:
            kwargs['fixed_ips'] = [{'ip_address': parsed_args.fixed_ip_address}]
        if parsed_args.tag:
            kwargs['tag'] = parsed_args.tag
        interface = compute_client.create_server_interface(server.id, **kwargs)
        columns = ('port_id', 'server_id', 'net_id', 'mac_addr', 'port_state', 'fixed_ips')
        column_headers = ('Port ID', 'Server ID', 'Network ID', 'MAC Address', 'Port State', 'Fixed IPs')
        if parsed_args.tag:
            if sdk_utils.supports_microversion(compute_client, '2.49'):
                columns += ('tag',)
                column_headers += ('Tag',)
        return (column_headers, utils.get_item_properties(interface, columns, formatters={'fixed_ips': format_columns.ListDictColumn}))