import json
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
class CreateDatabaseConfiguration(command.ShowOne):
    _description = _('Creates a configuration group.')

    def get_parser(self, prog_name):
        parser = super(CreateDatabaseConfiguration, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('Name of the configuration group.'))
        parser.add_argument('values', metavar='<values>', help=_('Dictionary of the values to set.'))
        parser.add_argument('--datastore', metavar='<datastore>', default=None, help=_('Datastore assigned to the configuration group. Required if default datastore is not configured.'))
        parser.add_argument('--datastore-version', metavar='<datastore_version>', default=None, help=_('Datastore version ID assigned to the configuration group.'))
        parser.add_argument('--datastore-version-number', default=None, help=_('The version number for the database. The version number is needed for the datastore versions with the same name.'))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('An optional description for the configuration group.'))
        return parser

    def take_action(self, parsed_args):
        db_configurations = self.app.client_manager.database.configurations
        config_grp = db_configurations.create(parsed_args.name, parsed_args.values, description=parsed_args.description, datastore=parsed_args.datastore, datastore_version=parsed_args.datastore_version, datastore_version_number=parsed_args.datastore_version_number)
        config_grp = set_attributes_for_print_detail(config_grp)
        return zip(*sorted(config_grp.items()))