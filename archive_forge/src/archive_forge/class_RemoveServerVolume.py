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
class RemoveServerVolume(command.Command):
    _description = _('Remove volume from server.\n\nSpecify ``--os-compute-api-version 2.20`` or higher to remove a\nvolume from a server with status ``SHELVED`` or ``SHELVED_OFFLOADED``.')

    def get_parser(self, prog_name):
        parser = super(RemoveServerVolume, self).get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', help=_('Server (name or ID)'))
        parser.add_argument('volume', metavar='<volume>', help=_('Volume to remove (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        volume_client = self.app.client_manager.sdk_connection.volume
        server = compute_client.find_server(parsed_args.server, ignore_missing=False)
        volume = volume_client.find_volume(parsed_args.volume, ignore_missing=False)
        compute_client.delete_volume_attachment(volume, server, ignore_missing=False)