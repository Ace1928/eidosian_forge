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
class ListDatabaseClusterModules(command.Lister):
    _description = _('Lists all modules for each instance of a cluster.')
    columns = ['instance_name', 'Module Name', 'Module Type', 'md5', 'created', 'updated']

    def get_parser(self, prog_name):
        parser = super(ListDatabaseClusterModules, self).get_parser(prog_name)
        parser.add_argument('cluster', metavar='<cluster>', help=_('ID or name of the cluster.'))
        return parser

    def take_action(self, parsed_args):
        database_clusters = self.app.client_manager.database.clusters
        database_instances = self.app.client_manager.database.instances
        cluster = utils.find_resource(database_clusters, parsed_args.cluster)
        instances = cluster._info['instances']
        modules = []
        for instance in instances:
            new_list = database_instances.modules(instance['id'])
            for item in new_list:
                item.instance_id = instance['id']
                item.instance_name = instance['name']
                item.module_name = item.name
                item.module_type = item.type
            modules += new_list
        modules = [utils.get_item_properties(module, self.columns) for module in modules]
        return (self.columns, modules)