from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from troveclient.i18n import _
from troveclient.v1.shell import _parse_extended_properties
from troveclient.v1.shell import _parse_instance_options
from troveclient.v1.shell import EXT_PROPS_HELP
from troveclient.v1.shell import EXT_PROPS_METAVAR
from troveclient.v1.shell import INSTANCE_HELP
from troveclient.v1.shell import INSTANCE_METAVAR
class ShowDatabaseCluster(command.ShowOne):
    _description = _('Shows details of a database cluster')

    def get_parser(self, prog_name):
        parser = super(ShowDatabaseCluster, self).get_parser(prog_name)
        parser.add_argument('cluster', metavar='<cluster>', help=_('ID or name of the cluster'))
        return parser

    def take_action(self, parsed_args):
        database_clusters = self.app.client_manager.database.clusters
        cluster = utils.find_resource(database_clusters, parsed_args.cluster)
        cluster = set_attributes_for_print_detail(cluster)
        return zip(*sorted(cluster.items()))