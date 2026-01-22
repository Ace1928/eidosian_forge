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
class DeleteServer(command.Command):
    _description = _('Delete server(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteServer, self).get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', nargs='+', help=_('Server(s) to delete (name or ID)'))
        parser.add_argument('--force', action='store_true', help=_('Force delete server(s)'))
        parser.add_argument('--all-projects', action='store_true', default=boolenv('ALL_PROJECTS'), help=_('Delete server(s) in another project by name (admin only)(can be specified using the ALL_PROJECTS envvar)'))
        parser.add_argument('--wait', action='store_true', help=_('Wait for delete to complete'))
        return parser

    def take_action(self, parsed_args):

        def _show_progress(progress):
            if progress:
                self.app.stdout.write('\rProgress: %s' % progress)
                self.app.stdout.flush()
        compute_client = self.app.client_manager.compute
        for server in parsed_args.server:
            server_obj = utils.find_resource(compute_client.servers, server, all_tenants=parsed_args.all_projects)
            if parsed_args.force:
                compute_client.servers.force_delete(server_obj.id)
            else:
                compute_client.servers.delete(server_obj.id)
            if parsed_args.wait:
                if not utils.wait_for_delete(compute_client.servers, server_obj.id, callback=_show_progress):
                    msg = _('Error deleting server: %s') % server_obj.id
                    raise exceptions.CommandError(msg)