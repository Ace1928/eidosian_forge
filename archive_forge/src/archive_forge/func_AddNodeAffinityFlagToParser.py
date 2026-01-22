from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddNodeAffinityFlagToParser(parser, is_update=False):
    """Adds a node affinity flag used for scheduling instances."""
    sole_tenancy_group = parser.add_group('Sole Tenancy.', mutex=True)
    sole_tenancy_group.add_argument('--node-affinity-file', type=arg_parsers.FileContents(), help="          The JSON/YAML file containing the configuration of desired nodes onto\n          which this instance could be scheduled. These rules filter the nodes\n          according to their node affinity labels. A node's affinity labels come\n          from the node template of the group the node is in.\n\n          The file should contain a list of a JSON/YAML objects. For an example,\n          see https://cloud.google.com/compute/docs/nodes/provisioning-sole-tenant-vms#configure_node_affinity_labels.\n          The following list describes the fields:\n\n          *key*::: Corresponds to the node affinity label keys of\n          the Node resource.\n          *operator*::: Specifies the node selection type. Must be one of:\n            `IN`: Requires Compute Engine to seek for matched nodes.\n            `NOT_IN`: Requires Compute Engine to avoid certain nodes.\n          *values*::: Optional. A list of values which correspond to the node\n          affinity label values of the Node resource.\n          ")
    sole_tenancy_group.add_argument('--node-group', help='The name of the node group to schedule this instance on.')
    sole_tenancy_group.add_argument('--node', help='The name of the node to schedule this instance on.')
    if is_update:
        sole_tenancy_group.add_argument('--clear-node-affinities', action='store_true', help='          Removes the node affinities field from the instance. If specified,\n          the instance node settings will be cleared. The instance will not be\n          scheduled onto a sole-tenant node.\n          ')