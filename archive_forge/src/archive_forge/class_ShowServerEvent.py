import logging
import uuid
from cliff import columns
import iso8601
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
class ShowServerEvent(command.ShowOne):
    """Show server event details.

    Specify ``--os-compute-api-version 2.21`` or higher to show event details
    for a deleted server, specified by ID only. Specify
    ``--os-compute-api-version 2.51`` or higher to show event details for
    non-admin users.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', help=_('Server to show event details (name or ID)'))
        parser.add_argument('request_id', metavar='<request-id>', help=_('Request ID of the event to show (ID only)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        try:
            server_id = compute_client.find_server(parsed_args.server, ignore_missing=False).id
        except sdk_exceptions.ResourceNotFound:
            if is_uuid_like(parsed_args.server):
                server_id = parsed_args.server
            else:
                raise
        server_action = compute_client.get_server_action(parsed_args.request_id, server_id)
        column_headers, columns = _get_server_event_columns(server_action, compute_client)
        return (column_headers, utils.get_item_properties(server_action, columns, formatters={'events': ServerActionEventColumn}))