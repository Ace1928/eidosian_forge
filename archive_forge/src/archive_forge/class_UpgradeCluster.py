import os
from magnumclient.common import utils as magnum_utils
from magnumclient import exceptions
from magnumclient.i18n import _
from osc_lib.command import command
from osc_lib import utils
class UpgradeCluster(command.Command):
    _description = _('Upgrade a Cluster')

    def get_parser(self, prog_name):
        parser = super(UpgradeCluster, self).get_parser(prog_name)
        parser.add_argument('cluster', metavar='<cluster>', help=_('The name or UUID of cluster to update'))
        parser.add_argument('cluster_template', help=_('The new cluster template ID will be upgraded to.'))
        parser.add_argument('--max-batch-size', metavar='<max_batch_size>', type=int, default=1, help=_('The max batch size for upgrading each time.'))
        parser.add_argument('--nodegroup', metavar='<nodegroup>', help=_('The name or UUID of the nodegroup of current cluster.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        mag_client = self.app.client_manager.container_infra
        cluster = mag_client.clusters.get(parsed_args.cluster)
        mag_client.clusters.upgrade(cluster.uuid, parsed_args.cluster_template, parsed_args.max_batch_size, parsed_args.nodegroup)
        print('Request to upgrade cluster %s has been accepted.' % parsed_args.cluster)