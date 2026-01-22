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
class UnsetVolumeSnapshot(command.Command):
    _description = _('Unset volume snapshot properties')

    def get_parser(self, prog_name):
        parser = super(UnsetVolumeSnapshot, self).get_parser(prog_name)
        parser.add_argument('snapshot', metavar='<snapshot>', help=_('Snapshot to modify (name or ID)'))
        parser.add_argument('--property', metavar='<key>', action='append', default=[], help=_('Property to remove from snapshot (repeat option to remove multiple properties)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        snapshot = utils.find_resource(volume_client.volume_snapshots, parsed_args.snapshot)
        if parsed_args.property:
            volume_client.volume_snapshots.delete_metadata(snapshot.id, parsed_args.property)