import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import exceptions as apiclient_exceptions
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import cliutils
from manilaclient.osc import utils
class ShareMigrationStart(command.Command):
    """Migrates share to a new host (Admin only, Experimental)."""
    _description = _('Migrates share to a new host.')

    def get_parser(self, prog_name):
        parser = super(ShareMigrationStart, self).get_parser(prog_name)
        parser.add_argument('share', metavar='<share>', help=_('Name or ID of share to migrate.'))
        parser.add_argument('host', metavar='<host>', help=_("Destination host where share will be migrated to. Use the format 'host@backend#pool'."))
        parser.add_argument('--force-host-assisted-migration', metavar='<force-host-assisted-migration>', choices=['True', 'False'], default=False, help=_('Enforces the use of the host-assisted migration approach, which bypasses driver optimizations. Default=False.'))
        parser.add_argument('--preserve-metadata', metavar='<preserve-metadata>', required=True, choices=['True', 'False'], help=_('Enforces migration to preserve all file metadata when moving its contents. If set to True, host-assistedmigration will not be attempted.'))
        parser.add_argument('--preserve-snapshots', metavar='<preserve-snapshots>', required=True, choices=['True', 'False'], help=_('Enforces migration of the share snapshots to the destination. If set to True, host-assisted migrationwill not be attempted.'))
        parser.add_argument('--writable', metavar='<writable>', required=True, choices=['True', 'False'], help=_('Enforces migration to keep the share writable while contents are being moved. If set to True, host-assistedmigration will not be attempted.'))
        parser.add_argument('--nondisruptive', metavar='<nondisruptive>', choices=['True', 'False'], required=True, help=_('Enforces migration to be nondisruptive. If set to True, host-assisted migration will not be attempted.'))
        parser.add_argument('--new-share-network', metavar='<new_share_network>', default=None, help=_('Specify the new share network for the share. Do not specify this parameter if the migrating share has to beretained within its current share network.'))
        parser.add_argument('--new-share-type', metavar='<new-share-type>', default=None, help=_('Specify the new share type for the share. Do not specify this parameter if the migrating share has to be retained with its current share type.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        new_share_net_id = None
        if parsed_args.new_share_network:
            new_share_net_id = apiutils.find_resource(share_client.share_networks, parsed_args.new_share_network).id
        new_share_type_id = None
        if parsed_args.new_share_type:
            new_share_type_id = apiutils.find_resource(share_client.share_types, parsed_args.new_share_type).id
        share = apiutils.find_resource(share_client.shares, parsed_args.share)
        share.migration_start(parsed_args.host, parsed_args.force_host_assisted_migration, parsed_args.preserve_metadata, parsed_args.writable, parsed_args.nondisruptive, parsed_args.preserve_snapshots, new_share_net_id, new_share_type_id)