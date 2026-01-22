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
class ListServerEvent(command.Lister):
    """List recent events of a server.

    Specify ``--os-compute-api-version 2.21`` or higher to show events for a
    deleted server, specified by ID only.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', help=_('Server to list events (name or ID)'))
        parser.add_argument('--long', action='store_true', default=False, help=_('List additional fields in output'))
        parser.add_argument('--changes-since', dest='changes_since', metavar='<changes-since>', help=_('List only server events changed later or equal to a certain point of time. The provided time should be an ISO 8061 formatted time, e.g. ``2016-03-04T06:27:59Z``. (supported with --os-compute-api-version 2.58 or above)'))
        parser.add_argument('--changes-before', dest='changes_before', metavar='<changes-before>', help=_('List only server events changed earlier or equal to a certain point of time. The provided time should be an ISO 8061 formatted time, e.g. ``2016-03-04T06:27:59Z``. (supported with --os-compute-api-version 2.66 or above)'))
        pagination.add_marker_pagination_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        kwargs = {}
        if parsed_args.marker:
            if not sdk_utils.supports_microversion(compute_client, '2.58'):
                msg = _('--os-compute-api-version 2.58 or greater is required to support the --marker option')
                raise exceptions.CommandError(msg)
            kwargs['marker'] = parsed_args.marker
        if parsed_args.limit:
            if not sdk_utils.supports_microversion(compute_client, '2.58'):
                msg = _('--os-compute-api-version 2.58 or greater is required to support the --limit option')
                raise exceptions.CommandError(msg)
            kwargs['limit'] = parsed_args.limit
            kwargs['paginated'] = False
        if parsed_args.changes_since:
            if not sdk_utils.supports_microversion(compute_client, '2.58'):
                msg = _('--os-compute-api-version 2.58 or greater is required to support the --changes-since option')
                raise exceptions.CommandError(msg)
            try:
                iso8601.parse_date(parsed_args.changes_since)
            except (TypeError, iso8601.ParseError):
                msg = _('Invalid changes-since value: %s')
                raise exceptions.CommandError(msg % parsed_args.changes_since)
            kwargs['changes_since'] = parsed_args.changes_since
        if parsed_args.changes_before:
            if not sdk_utils.supports_microversion(compute_client, '2.66'):
                msg = _('--os-compute-api-version 2.66 or greater is required to support the --changes-before option')
                raise exceptions.CommandError(msg)
            try:
                iso8601.parse_date(parsed_args.changes_before)
            except (TypeError, iso8601.ParseError):
                msg = _('Invalid changes-before value: %s')
                raise exceptions.CommandError(msg % parsed_args.changes_before)
            kwargs['changes_before'] = parsed_args.changes_before
        try:
            server_id = compute_client.find_server(parsed_args.server, ignore_missing=False).id
        except sdk_exceptions.ResourceNotFound:
            if is_uuid_like(parsed_args.server):
                server_id = parsed_args.server
            else:
                raise
        data = compute_client.server_actions(server_id, **kwargs)
        columns = ('request_id', 'server_id', 'action', 'start_time')
        column_headers = ('Request ID', 'Server ID', 'Action', 'Start Time')
        if parsed_args.long:
            columns += ('message', 'project_id', 'user_id')
            column_headers += ('Message', 'Project ID', 'User ID')
        return (column_headers, (utils.get_item_properties(s, columns) for s in data))