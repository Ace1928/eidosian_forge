import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
class ShowShareGroupSnapshot(command.ShowOne):
    """Display a share group snapshot"""
    _description = _('Show details about a share group snapshot')

    def get_parser(self, prog_name):
        parser = super(ShowShareGroupSnapshot, self).get_parser(prog_name)
        parser.add_argument('share_group_snapshot', metavar='<share-group-snapshot>', help=_('Name or ID of the share group snapshot to display'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_group_snapshot = osc_utils.find_resource(share_client.share_group_snapshots, parsed_args.share_group_snapshot)
        data = share_group_snapshot._info
        data.pop('links', None)
        data.pop('members', None)
        return self.dict2columns(data)