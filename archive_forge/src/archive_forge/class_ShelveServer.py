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
class ShelveServer(command.Command):
    """Shelve and optionally offload server(s).

    Shelving a server creates a snapshot of the server and stores this
    snapshot before shutting down the server. This shelved server can then be
    offloaded or deleted from the host, freeing up remaining resources on the
    host, such as network interfaces. Shelved servers can be unshelved,
    restoring the server from the snapshot. Shelving is therefore useful where
    users wish to retain the UUID and IP of a server, without utilizing other
    resources or disks.

    Most clouds are configured to automatically offload shelved servers
    immediately or after a small delay. For clouds where this is not
    configured, or where the delay is larger, offloading can be manually
    specified. This is an admin-only operation by default.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('servers', metavar='<server>', nargs='+', help=_('Server(s) to shelve (name or ID)'))
        parser.add_argument('--offload', action='store_true', default=False, help=_('Remove the shelved server(s) from the host (admin only). Invoking this option on an unshelved server(s) will result in the server being shelved first'))
        parser.add_argument('--wait', action='store_true', default=False, help=_('Wait for shelve and/or offload operation to complete'))
        return parser

    def take_action(self, parsed_args):

        def _show_progress(progress):
            if progress:
                self.app.stdout.write('\rProgress: %s' % progress)
                self.app.stdout.flush()
        compute_client = self.app.client_manager.sdk_connection.compute
        server_ids = []
        for server in parsed_args.servers:
            server_obj = compute_client.find_server(server, ignore_missing=False)
            if server_obj.status.lower() in ('shelved', 'shelved_offloaded'):
                continue
            server_ids.append(server_obj.id)
            compute_client.shelve_server(server_obj.id)
        if not parsed_args.wait and (not parsed_args.offload):
            return
        for server_id in server_ids:
            if not utils.wait_for_status(compute_client.get_server, server_id, success_status=('shelved', 'shelved_offloaded'), callback=_show_progress):
                msg = _('Error shelving server: %s') % server_id
                raise exceptions.CommandError(msg)
        if not parsed_args.offload:
            return
        for server_id in server_ids:
            server_obj = compute_client.get_server(server_id)
            if server_obj.status.lower() == 'shelved_offloaded':
                continue
            compute_client.shelve_offload_server(server_id)
        if not parsed_args.wait:
            return
        for server_id in server_ids:
            if not utils.wait_for_status(compute_client.get_server, server_id, success_status=('shelved_offloaded',), callback=_show_progress):
                msg = _('Error offloading shelved server: %s') % server_id
                raise exceptions.CommandError(msg)