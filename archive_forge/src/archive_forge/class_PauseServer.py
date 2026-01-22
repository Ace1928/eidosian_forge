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
class PauseServer(command.Command):
    _description = _('Pause server(s)')

    def get_parser(self, prog_name):
        parser = super(PauseServer, self).get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', nargs='+', help=_('Server(s) to pause (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        for server in parsed_args.server:
            server_id = compute_client.find_server(server, ignore_missing=False).id
            compute_client.pause_server(server_id)