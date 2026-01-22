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
class SetVolumeSnapshot(command.Command):
    _description = _('Set volume snapshot properties')

    def get_parser(self, prog_name):
        parser = super(SetVolumeSnapshot, self).get_parser(prog_name)
        parser.add_argument('snapshot', metavar='<snapshot>', help=_('Snapshot to modify (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('New snapshot name'))
        parser.add_argument('--description', metavar='<description>', help=_('New snapshot description'))
        parser.add_argument('--no-property', dest='no_property', action='store_true', help=_('Remove all properties from <snapshot> (specify both --no-property and --property to remove the current properties before setting new properties.)'))
        parser.add_argument('--property', metavar='<key=value>', action=parseractions.KeyValueAction, help=_('Property to add/change for this snapshot (repeat option to set multiple properties)'))
        parser.add_argument('--state', metavar='<state>', choices=['available', 'error', 'creating', 'deleting', 'error_deleting'], help=_('New snapshot state. ("available", "error", "creating", "deleting", or "error_deleting") (admin only) (This option simply changes the state of the snapshot in the database with no regard to actual status, exercise caution when using)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        snapshot = utils.find_resource(volume_client.volume_snapshots, parsed_args.snapshot)
        result = 0
        if parsed_args.no_property:
            try:
                key_list = snapshot.metadata.keys()
                volume_client.volume_snapshots.delete_metadata(snapshot.id, list(key_list))
            except Exception as e:
                LOG.error(_('Failed to clean snapshot properties: %s'), e)
                result += 1
        if parsed_args.property:
            try:
                volume_client.volume_snapshots.set_metadata(snapshot.id, parsed_args.property)
            except Exception as e:
                LOG.error(_('Failed to set snapshot property: %s'), e)
                result += 1
        if parsed_args.state:
            try:
                volume_client.volume_snapshots.reset_state(snapshot.id, parsed_args.state)
            except Exception as e:
                LOG.error(_('Failed to set snapshot state: %s'), e)
                result += 1
        kwargs = {}
        if parsed_args.name:
            kwargs['name'] = parsed_args.name
        if parsed_args.description:
            kwargs['description'] = parsed_args.description
        if kwargs:
            try:
                volume_client.volume_snapshots.update(snapshot.id, **kwargs)
            except Exception as e:
                LOG.error(_('Failed to update snapshot name or description: %s'), e)
                result += 1
        if result > 0:
            raise exceptions.CommandError(_('One or more of the set operations failed'))