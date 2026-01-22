import json
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
class AttachDatabaseConfiguration(command.Command):
    _description = _('Attaches a configuration group to an instance.')

    def get_parser(self, prog_name):
        parser = super(AttachDatabaseConfiguration, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', type=str, help=_('ID or name of the instance'))
        parser.add_argument('configuration', metavar='<configuration>', type=str, help=_('ID or name of the configuration group to attach to the instance.'))
        return parser

    def take_action(self, parsed_args):
        manager = self.app.client_manager.database
        db_instances = manager.instances
        db_configurations = manager.configurations
        instance_id = parsed_args.instance
        config_id = parsed_args.configuration
        if not uuidutils.is_uuid_like(instance_id):
            instance_id = osc_utils.find_resource(db_instances, instance_id)
        if not uuidutils.is_uuid_like(config_id):
            config_id = osc_utils.find_resource(db_configurations, config_id)
        db_instances.update(instance_id, configuration=config_id)