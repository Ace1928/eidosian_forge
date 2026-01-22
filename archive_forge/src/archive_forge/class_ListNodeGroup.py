from magnumclient.common import utils as magnum_utils
from magnumclient.i18n import _
from osc_lib.command import command
from osc_lib import utils
class ListNodeGroup(command.Lister):
    _description = _('List nodegroups')

    def get_parser(self, prog_name):
        parser = super(ListNodeGroup, self).get_parser(prog_name)
        parser.add_argument('cluster', metavar='<cluster>', help=_('ID or name of the cluster where the nodegroup belongs.'))
        parser.add_argument('--limit', metavar='<limit>', type=int, help=_('Maximum number of nodegroups to return'))
        parser.add_argument('--sort-key', metavar='<sort-key>', help=_('Column to sort results by'))
        parser.add_argument('--sort-dir', metavar='<sort-dir>', choices=['desc', 'asc'], help=_('Direction to sort. "asc" or "desc".'))
        parser.add_argument('--role', metavar='<role>', help=_('List the nodegroups in the cluster with this role'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        mag_client = self.app.client_manager.container_infra
        columns = ['uuid', 'name', 'flavor_id', 'image_id', 'node_count', 'status', 'role']
        cluster_id = parsed_args.cluster
        nodegroups = mag_client.nodegroups.list(cluster_id, limit=parsed_args.limit, sort_key=parsed_args.sort_key, sort_dir=parsed_args.sort_dir, role=parsed_args.role)
        return (columns, (utils.get_item_properties(n, columns) for n in nodegroups))