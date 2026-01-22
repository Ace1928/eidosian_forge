import json
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
class ShowDatabaseConfigurationParameter(command.ShowOne):
    _description = _('Shows details of a database configuration parameter.')

    def get_parser(self, prog_name):
        parser = super(ShowDatabaseConfigurationParameter, self).get_parser(prog_name)
        parser.add_argument('datastore_version', metavar='<datastore_version>', help=_('Datastore version name or ID assigned to the configuration group. ID is preferred if more than one datastore versions have the same name.'))
        parser.add_argument('parameter', metavar='<parameter>', help=_('Name of the configuration parameter.'))
        parser.add_argument('--datastore', metavar='<datastore>', default=None, help=_('ID or name of the datastore to list configuration parameters for. Optional if the ID of the datastore_version is provided.'))
        return parser

    def take_action(self, parsed_args):
        db_configuration_parameters = self.app.client_manager.database.configuration_parameters
        if uuidutils.is_uuid_like(parsed_args.datastore_version):
            param = db_configuration_parameters.get_parameter_by_version(parsed_args.datastore_version, parsed_args.parameter)
        elif parsed_args.datastore:
            param = db_configuration_parameters.get_parameter(parsed_args.datastore, parsed_args.datastore_version, parsed_args.parameter)
        else:
            raise exceptions.NoUniqueMatch(_('Either datastore version ID or datastore name needs to be specified.'))
        return zip(*sorted(param._info.items()))