import copy
import functools
import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class ShowVolumeSnapshot(command.ShowOne):
    _description = _('Display volume snapshot details')

    def get_parser(self, prog_name):
        parser = super(ShowVolumeSnapshot, self).get_parser(prog_name)
        parser.add_argument('snapshot', metavar='<snapshot>', help=_('Snapshot to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        snapshot = utils.find_resource(volume_client.volume_snapshots, parsed_args.snapshot)
        snapshot._info.update({'properties': format_columns.DictColumn(snapshot._info.pop('metadata'))})
        return zip(*sorted(snapshot._info.items()))