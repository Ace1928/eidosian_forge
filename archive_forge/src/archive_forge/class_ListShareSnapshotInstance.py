from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from manilaclient.common._i18n import _
from manilaclient.common import cliutils
class ListShareSnapshotInstance(command.Lister):
    """List all share snapshot instances."""
    _description = _('List all share snapshot instances')

    def get_parser(self, prog_name):
        parser = super(ListShareSnapshotInstance, self).get_parser(prog_name)
        parser.add_argument('--snapshot', metavar='<snapshot>', default=None, help=_('Filter results by share snapshot ID.'))
        parser.add_argument('--detailed', action='store_true', help=_('Show detailed information about snapshot instances. '))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        snapshot = share_client.share_snapshots.get(parsed_args.snapshot) if parsed_args.snapshot else None
        share_snapshot_instances = share_client.share_snapshot_instances.list(detailed=parsed_args.detailed, snapshot=snapshot)
        list_of_keys = ['ID', 'Snapshot ID', 'Status']
        if parsed_args.detailed:
            list_of_keys += ['Created At', 'Updated At', 'Share ID', 'Share Instance ID', 'Progress', 'Provider Location']
        return (list_of_keys, (utils.get_item_properties(s, list_of_keys) for s in share_snapshot_instances))