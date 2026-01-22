import uuid
from openstack import utils as sdk_utils
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class AbortMigration(command.Command):
    """Cancel an ongoing live migration.

    This command requires ``--os-compute-api-version`` 2.24 or greater.
    """

    def get_parser(self, prog_name):
        parser = super(AbortMigration, self).get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', help=_('Server (name or ID)'))
        parser.add_argument('migration', metavar='<migration>', help=_('Migration (ID)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        if not sdk_utils.supports_microversion(compute_client, '2.24'):
            msg = _('--os-compute-api-version 2.24 or greater is required to support the server migration abort command')
            raise exceptions.CommandError(msg)
        if not parsed_args.migration.isdigit():
            try:
                uuid.UUID(parsed_args.migration)
            except ValueError:
                msg = _('The <migration> argument must be an ID or UUID')
                raise exceptions.CommandError(msg)
            if not sdk_utils.supports_microversion(compute_client, '2.59'):
                msg = _('--os-compute-api-version 2.59 or greater is required to abort server migrations by UUID')
                raise exceptions.CommandError(msg)
        server = compute_client.find_server(parsed_args.server, ignore_missing=False)
        migration_id = parsed_args.migration
        if not parsed_args.migration.isdigit():
            migration_id = _get_migration_by_uuid(compute_client, server.id, parsed_args.migration).id
        compute_client.abort_server_migration(migration_id, server.id, ignore_missing=False)