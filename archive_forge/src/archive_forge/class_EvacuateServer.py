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
class EvacuateServer(command.ShowOne):
    _description = _('Evacuate a server to a different host.\n\nThis command is used to recreate a server after the host it was on has failed.\nIt can only be used if the compute service that manages the server is down.\nThis command should only be used by an admin after they have confirmed that the\ninstance is not running on the failed host.\n\nIf the server instance was created with an ephemeral root disk on non-shared\nstorage the server will be rebuilt using the original glance image preserving\nthe ports and any attached data volumes.\n\nIf the server uses boot for volume or has its root disk on shared storage the\nroot disk will be preserved and reused for the evacuated instance on the new\nhost.')

    def get_parser(self, prog_name):
        parser = super(EvacuateServer, self).get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', help=_('Server (name or ID)'))
        parser.add_argument('--wait', action='store_true', help=_('Wait for evacuation to complete'))
        parser.add_argument('--host', metavar='<host>', default=None, help=_('Set the preferred host on which to rebuild the evacuated server. The host will be validated by the scheduler. (supported by --os-compute-api-version 2.29 or above)'))
        shared_storage_group = parser.add_mutually_exclusive_group()
        shared_storage_group.add_argument('--password', metavar='<password>', default=None, help=_('Set the password on the evacuated instance. This option is mutually exclusive with the --shared-storage option. This option requires cloud support.'))
        shared_storage_group.add_argument('--shared-storage', action='store_true', dest='shared_storage', help=_('Indicate that the instance is on shared storage. This will be auto-calculated with --os-compute-api-version 2.14 and greater and should not be used with later microversions. This option is mutually exclusive with the --password option'))
        return parser

    def take_action(self, parsed_args):

        def _show_progress(progress):
            if progress:
                self.app.stdout.write('\rProgress: %s' % progress)
                self.app.stdout.flush()
        compute_client = self.app.client_manager.compute
        image_client = self.app.client_manager.image
        if parsed_args.host:
            if compute_client.api_version < api_versions.APIVersion('2.29'):
                msg = _('--os-compute-api-version 2.29 or later is required to specify a preferred host.')
                raise exceptions.CommandError(msg)
        if parsed_args.shared_storage:
            if compute_client.api_version > api_versions.APIVersion('2.13'):
                msg = _('--os-compute-api-version 2.13 or earlier is required to specify shared-storage.')
                raise exceptions.CommandError(msg)
        kwargs = {'host': parsed_args.host, 'password': parsed_args.password}
        if compute_client.api_version <= api_versions.APIVersion('2.13'):
            kwargs['on_shared_storage'] = parsed_args.shared_storage
        server = utils.find_resource(compute_client.servers, parsed_args.server)
        server.evacuate(**kwargs)
        if parsed_args.wait:
            if utils.wait_for_status(compute_client.servers.get, server.id, callback=_show_progress):
                self.app.stdout.write(_('Complete\n'))
            else:
                msg = _('Error evacuating server: %s') % server.id
                raise exceptions.CommandError(msg)
        details = _prep_server_detail(compute_client, image_client, server, refresh=True)
        return zip(*sorted(details.items()))