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
class ShareSnapshotAccessList(command.Lister):
    """Show access list for a snapshot"""
    _description = _('Show access list for a snapshot')

    def get_parser(self, prog_name):
        parser = super(ShareSnapshotAccessList, self).get_parser(prog_name)
        parser.add_argument('snapshot', metavar='<snapshot>', help=_('Name or ID of the share snapshot to show access list for.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        snapshot_obj = utils.find_resource(share_client.share_snapshots, parsed_args.snapshot)
        access_rules = share_client.share_snapshots.access_list(snapshot_obj)
        columns = ['ID', 'Access Type', 'Access To', 'State']
        return (columns, (utils.get_item_properties(s, columns) for s in access_rules))