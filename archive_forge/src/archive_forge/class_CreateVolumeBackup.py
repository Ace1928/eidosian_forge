import copy
import functools
import logging
from cinderclient import api_versions
from cliff import columns as cliff_columns
from openstack import utils as sdk_utils
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
class CreateVolumeBackup(command.ShowOne):
    _description = _('Create new volume backup')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('volume', metavar='<volume>', help=_('Volume to backup (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('Name of the backup'))
        parser.add_argument('--description', metavar='<description>', help=_('Description of the backup'))
        parser.add_argument('--container', metavar='<container>', help=_('Optional backup container name'))
        parser.add_argument('--snapshot', metavar='<snapshot>', help=_('Snapshot to backup (name or ID)'))
        parser.add_argument('--force', action='store_true', default=False, help=_('Allow to back up an in-use volume'))
        parser.add_argument('--incremental', action='store_true', default=False, help=_('Perform an incremental backup'))
        parser.add_argument('--no-incremental', action='store_false', help=_('Do not perform an incremental backup'))
        parser.add_argument('--property', metavar='<key=value>', action=parseractions.KeyValueAction, dest='properties', help=_('Set a property on this backup (repeat option to remove multiple values) (supported by --os-volume-api-version 3.43 or above)'))
        parser.add_argument('--availability-zone', metavar='<zone-name>', help=_('AZ where the backup should be stored; by default it will be the same as the source (supported by --os-volume-api-version 3.51 or above)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.sdk_connection.volume
        volume_id = volume_client.find_volume(parsed_args.volume, ignore_missing=False).id
        kwargs = {}
        if parsed_args.snapshot:
            kwargs['snapshot_id'] = volume_client.find_snapshot(parsed_args.snapshot, ignore_missing=False).id
        if parsed_args.properties:
            if not sdk_utils.supports_microversion(volume_client, '3.43'):
                msg = _('--os-volume-api-version 3.43 or greater is required to support the --property option')
                raise exceptions.CommandError(msg)
            kwargs['metadata'] = parsed_args.properties
        if parsed_args.availability_zone:
            if not sdk_utils.supports_microversion(volume_client, '3.51'):
                msg = _('--os-volume-api-version 3.51 or greater is required to support the --availability-zone option')
                raise exceptions.CommandError(msg)
            kwargs['availability_zone'] = parsed_args.availability_zone
        columns = ('id', 'name', 'volume_id')
        backup = volume_client.create_backup(volume_id=volume_id, container=parsed_args.container, name=parsed_args.name, description=parsed_args.description, force=parsed_args.force, incremental=parsed_args.incremental, **kwargs)
        data = utils.get_dict_properties(backup, columns)
        return (columns, data)