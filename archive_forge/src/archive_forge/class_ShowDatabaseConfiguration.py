import json
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
class ShowDatabaseConfiguration(command.ShowOne):
    _description = _('Shows details of a database configuration group.')

    def get_parser(self, prog_name):
        parser = super(ShowDatabaseConfiguration, self).get_parser(prog_name)
        parser.add_argument('configuration_group', metavar='<configuration_group>', help=_('ID or name of the configuration group'))
        return parser

    def take_action(self, parsed_args):
        db_configurations = self.app.client_manager.database.configurations
        configuration = osc_utils.find_resource(db_configurations, parsed_args.configuration_group)
        configuration = set_attributes_for_print_detail(configuration)
        return zip(*sorted(configuration.items()))