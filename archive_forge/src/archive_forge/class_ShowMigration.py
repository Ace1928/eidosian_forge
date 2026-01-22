import uuid
from openstack import utils as sdk_utils
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class ShowMigration(command.ShowOne):
    """Show an in-progress live migration for a given server.

    Note that it is not possible to show cold migrations or completed
    live-migrations. Use 'openstack server migration list' to get details for
    these.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', help=_('Server (name or ID)'))
        parser.add_argument('migration', metavar='<migration>', help=_('Migration (ID)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        if not sdk_utils.supports_microversion(compute_client, '2.24'):
            msg = _('--os-compute-api-version 2.24 or greater is required to support the server migration show command')
            raise exceptions.CommandError(msg)
        if not parsed_args.migration.isdigit():
            try:
                uuid.UUID(parsed_args.migration)
            except ValueError:
                msg = _('The <migration> argument must be an ID or UUID')
                raise exceptions.CommandError(msg)
            if not sdk_utils.supports_microversion(compute_client, '2.59'):
                msg = _('--os-compute-api-version 2.59 or greater is required to retrieve server migrations by UUID')
                raise exceptions.CommandError(msg)
        server = compute_client.find_server(parsed_args.server, ignore_missing=False)
        if not parsed_args.migration.isdigit():
            server_migration = _get_migration_by_uuid(compute_client, server.id, parsed_args.migration)
        else:
            server_migration = compute_client.get_server_migration(server.id, parsed_args.migration, ignore_missing=False)
        column_headers = ('ID', 'Server UUID', 'Status', 'Source Compute', 'Source Node', 'Dest Compute', 'Dest Host', 'Dest Node', 'Memory Total Bytes', 'Memory Processed Bytes', 'Memory Remaining Bytes', 'Disk Total Bytes', 'Disk Processed Bytes', 'Disk Remaining Bytes', 'Created At', 'Updated At')
        columns = ('id', 'server_id', 'status', 'source_compute', 'source_node', 'dest_compute', 'dest_host', 'dest_node', 'memory_total_bytes', 'memory_processed_bytes', 'memory_remaining_bytes', 'disk_total_bytes', 'disk_processed_bytes', 'disk_remaining_bytes', 'created_at', 'updated_at')
        if sdk_utils.supports_microversion(compute_client, '2.59'):
            column_headers += ('UUID',)
            columns += ('uuid',)
        if sdk_utils.supports_microversion(compute_client, '2.80'):
            column_headers += ('User ID', 'Project ID')
            columns += ('user_id', 'project_id')
        data = utils.get_item_properties(server_migration, columns)
        return (column_headers, data)