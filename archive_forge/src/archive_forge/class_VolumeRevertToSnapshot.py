import logging
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class VolumeRevertToSnapshot(command.Command):
    _description = _('Revert a volume to a snapshot.')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('snapshot', metavar='<snapshot>', help=_('Name or ID of the snapshot to restore. The snapshot must be the most recent one known to cinder.'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.sdk_connection.volume
        if not sdk_utils.supports_microversion(volume_client, '3.40'):
            msg = _("--os-volume-api-version 3.40 or greater is required to support the 'volume revert snapshot' command")
            raise exceptions.CommandError(msg)
        snapshot = volume_client.find_snapshot(parsed_args.snapshot, ignore_missing=False)
        volume = volume_client.find_volume(snapshot.volume_id, ignore_missing=False)
        volume_client.revert_volume_to_snapshot(volume, snapshot)