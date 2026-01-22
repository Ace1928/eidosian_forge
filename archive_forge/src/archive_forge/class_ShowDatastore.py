from osc_lib.command import command
from osc_lib import utils
from troveclient import exceptions
from troveclient.i18n import _
from troveclient import utils as tc_utils
class ShowDatastore(command.ShowOne):
    _description = _('Shows details of a datastore')

    def get_parser(self, prog_name):
        parser = super(ShowDatastore, self).get_parser(prog_name)
        parser.add_argument('datastore', metavar='<datastore>', help=_('ID of the datastore'))
        return parser

    def take_action(self, parsed_args):
        datastore_client = self.app.client_manager.database.datastores
        datastore = utils.find_resource(datastore_client, parsed_args.datastore)
        datastore = set_attributes_for_print_detail(datastore)
        return zip(*sorted(datastore.items()))