from magnumclient.common import utils as magnum_utils
from magnumclient.i18n import _
from osc_lib.command import command
from osc_lib import utils
class CreateNodeGroup(command.Command):
    _description = _('Create a nodegroup')

    def get_parser(self, prog_name):
        parser = super(CreateNodeGroup, self).get_parser(prog_name)
        parser.add_argument('--docker-volume-size', dest='docker_volume_size', type=int, metavar='<docker-volume-size>', help='The size in GB for the docker volume to use.')
        parser.add_argument('--labels', metavar='<KEY1=VALUE1,KEY2=VALUE2;KEY3=VALUE3...>', action='append', help=_('Arbitrary labels in the form of key=valuepairs to associate with a nodegroup. May be used multiple times.'))
        parser.add_argument('cluster', metavar='<cluster>', help='Name of the nodegroup to create.')
        parser.add_argument('name', metavar='<name>', help='Name of the nodegroup to create.')
        parser.add_argument('--node-count', dest='node_count', type=int, default=1, metavar='<node-count>', help='The nodegroup node count.')
        parser.add_argument('--min-nodes', dest='min_node_count', type=int, default=0, metavar='<min-nodes>', help='The nodegroup minimum node count.')
        parser.add_argument('--max-nodes', dest='max_node_count', type=int, default=None, metavar='<max-nodes>', help='The nodegroup maximum node count.')
        parser.add_argument('--role', dest='role', type=str, default='worker', metavar='<role>', help='The role of the nodegroup')
        parser.add_argument('--image', metavar='<image>', help=_('The name or UUID of the base image to customize for the NodeGroup.'))
        parser.add_argument('--flavor', metavar='<flavor>', help=_('The nova flavor name or UUID to use when launching the nodes in this NodeGroup.'))
        parser.add_argument('--merge-labels', dest='merge_labels', action='store_true', default=False, help=_('The labels provided will be merged with the labels configured in the specified cluster.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        mag_client = self.app.client_manager.container_infra
        args = {'name': parsed_args.name, 'node_count': parsed_args.node_count, 'max_node_count': parsed_args.max_node_count, 'min_node_count': parsed_args.min_node_count, 'role': parsed_args.role}
        if parsed_args.labels is not None:
            args['labels'] = magnum_utils.handle_labels(parsed_args.labels)
        if parsed_args.docker_volume_size is not None:
            args['docker_volume_size'] = parsed_args.docker_volume_size
        if parsed_args.flavor is not None:
            args['flavor_id'] = parsed_args.flavor
        if parsed_args.image is not None:
            args['image_id'] = parsed_args.image
        if parsed_args.merge_labels:
            args['merge_labels'] = parsed_args.merge_labels
        cluster_id = parsed_args.cluster
        nodegroup = mag_client.nodegroups.create(cluster_id, **args)
        print('Request to create nodegroup %s accepted' % nodegroup.uuid)