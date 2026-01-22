import json
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
class DeleteDatabaseConfiguration(command.Command):
    _description = _('Deletes a configuration group.')

    def get_parser(self, prog_name):
        parser = super(DeleteDatabaseConfiguration, self).get_parser(prog_name)
        parser.add_argument('configuration_group', metavar='<configuration_group>', help=_('ID or name of the configuration group'))
        return parser

    def take_action(self, parsed_args):
        db_configurations = self.app.client_manager.database.configurations
        try:
            configuration = osc_utils.find_resource(db_configurations, parsed_args.configuration_group)
            db_configurations.delete(configuration)
        except Exception as e:
            msg = _('Failed to delete configuration %(c_group)s: %(e)s') % {'c_group': parsed_args.configuration_group, 'e': e}
            raise exceptions.CommandError(msg)