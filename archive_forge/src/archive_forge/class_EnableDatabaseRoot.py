from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from troveclient.i18n import _
class EnableDatabaseRoot(command.ShowOne):
    _description = _('Enables root for an instance and resets if already exists.')

    def get_parser(self, prog_name):
        parser = super(EnableDatabaseRoot, self).get_parser(prog_name)
        parser.add_argument('instance_or_cluster', metavar='<instance_or_cluster>', help=_('ID or name of the instance or cluster.'))
        parser.add_argument('--root_password', metavar='<root_password>', default=None, help=_('Root password to set.'))
        return parser

    def take_action(self, parsed_args):
        database_client_manager = self.app.client_manager.database
        instance_or_cluster, resource_type = find_instance_or_cluster(database_client_manager, parsed_args.instance_or_cluster)
        db_root = database_client_manager.root
        if resource_type == 'instance':
            root = db_root.create_instance_root(instance_or_cluster, parsed_args.root_password)
        else:
            root = db_root.create_cluster_root(instance_or_cluster, parsed_args.root_password)
        result = {'name': root[0], 'password': root[1]}
        return zip(*sorted(result.items()))