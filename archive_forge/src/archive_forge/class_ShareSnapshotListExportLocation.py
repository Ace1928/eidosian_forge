import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common import cliutils
from manilaclient.osc import utils as oscutils
class ShareSnapshotListExportLocation(command.Lister):
    """List export locations of a given snapshot"""
    _description = _('List export locations of a given snapshot')

    def get_parser(self, prog_name):
        parser = super(ShareSnapshotListExportLocation, self).get_parser(prog_name)
        parser.add_argument('snapshot', metavar='<snapshot>', help=_('Name or ID of the share snapshot.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        snapshot_obj = utils.find_resource(share_client.share_snapshots, parsed_args.snapshot)
        export_locations = share_client.share_snapshot_export_locations.list(snapshot=snapshot_obj)
        columns = ['ID', 'Path']
        return (columns, (utils.get_item_properties(s, columns) for s in export_locations))