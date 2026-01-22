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
class ShareSnapshotAccessAllow(command.ShowOne):
    """Allow read only access to a snapshot."""
    _description = _('Allow access to a snapshot')

    def get_parser(self, prog_name):
        parser = super(ShareSnapshotAccessAllow, self).get_parser(prog_name)
        parser.add_argument('snapshot', metavar='<snapshot>', help=_('Name or ID of the snapshot'))
        parser.add_argument('access_type', metavar='<access_type>', help=_('Access rule type (only "ip", "user" (user or group), "cert" or "cephx" are supported).'))
        parser.add_argument('access_to', metavar='<access_to>', help=_('Value that defines access.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        snapshot_obj = utils.find_resource(share_client.share_snapshots, parsed_args.snapshot)
        try:
            snapshot_access = share_client.share_snapshots.allow(snapshot=snapshot_obj, access_type=parsed_args.access_type, access_to=parsed_args.access_to)
            return self.dict2columns(snapshot_access)
        except Exception as e:
            raise exceptions.CommandError("Failed to create access to share snapshot '%s': %s" % (snapshot_obj, e))