from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient.i18n import _
from troveclient.osc.v1 import base
from troveclient import utils as trove_utils
class CreateDatabaseBackup(command.ShowOne):
    _description = _('Creates a backup of an instance.')

    def get_parser(self, prog_name):
        parser = super(CreateDatabaseBackup, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('Name of the backup.'))
        parser.add_argument('-i', '--instance', metavar='<instance>', help=_('ID or name of the instance. This is not required if restoring a backup from the data location.'))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('An optional description for the backup.'))
        parser.add_argument('--parent', metavar='<parent>', default=None, help=_('Optional ID of the parent backup to perform an incremental backup from.'))
        parser.add_argument('--incremental', action='store_true', default=False, help=_('Create an incremental backup based on the last full or incremental backup. It will create a full backup if no existing backup found.'))
        parser.add_argument('--swift-container', help=_('The container name for storing the backup data when Swift is used as backup storage backend. If not specified, will use the container name configured in the backup strategy, otherwise, the default value configured by the cloud operator. Non-existent container is created automatically.'))
        parser.add_argument('--restore-from', help=_('The original backup data location, typically this is a Swift object URL.'))
        parser.add_argument('--restore-datastore-version', help=_('ID of the local datastore version corresponding to the original backup'))
        parser.add_argument('--restore-size', type=float, help=_('The original backup size.'))
        return parser

    def take_action(self, parsed_args):
        manager = self.app.client_manager.database
        database_backups = manager.backups
        params = {}
        instance_id = None
        if parsed_args.restore_from:
            params.update({'restore_from': parsed_args.restore_from, 'restore_ds_version': parsed_args.restore_datastore_version, 'restore_size': parsed_args.restore_size})
        elif not parsed_args.instance:
            raise exceptions.CommandError('Instance ID or name is required if not restoring a backup.')
        else:
            instance_id = trove_utils.get_resource_id(manager.instances, parsed_args.instance)
            params.update({'description': parsed_args.description, 'parent_id': parsed_args.parent, 'incremental': parsed_args.incremental, 'swift_container': parsed_args.swift_container})
        backup = database_backups.create(parsed_args.name, instance_id, **params)
        backup = set_attributes_for_print_detail(backup)
        return zip(*sorted(backup.items()))