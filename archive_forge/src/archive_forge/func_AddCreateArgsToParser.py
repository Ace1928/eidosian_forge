from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core.util import scaled_integer
import six
def AddCreateArgsToParser(parser):
    """Add flags for creating a node template to the argument parser."""
    parser.add_argument('--description', help='An optional description of this resource.')
    parser.add_argument('--node-affinity-labels', metavar='KEY=VALUE', type=arg_parsers.ArgDict(key_type=labels_util.KEY_FORMAT_VALIDATOR, value_type=labels_util.VALUE_FORMAT_VALIDATOR), action=arg_parsers.UpdateAction, help='Labels to use for node affinity, which will be used in instance scheduling. This corresponds to the `--node-affinity` flag on `compute instances create` and `compute instance-templates create`.')
    node_type_group = parser.add_group(mutex=True, required=True)
    node_type_group.add_argument('--node-type', help='          The node type to use for nodes in node groups using this template.\n          The type of a node determines what resources are available to\n          instances running on the node.\n\n          See the following for more information:\n\n              $ {grandparent_command} node-types list')
    node_type_group.add_argument('--node-requirements', type=arg_parsers.ArgDict(spec={'vCPU': _IntOrAny(), 'memory': _BinarySizeOrAny('MB'), 'localSSD': _BinarySizeOrAny('GB')}), help="The requirements for nodes. Google Compute Engine will automatically\nchoose a node type that fits the requirements on Node Group creation.\nIf multiple node types match your defined criteria, the NodeType with\nthe least amount of each resource will be selected. You can specify 'any'\nto indicate any non-zero value for a certain resource.\n\nThe following keys are allowed:\n\n*vCPU*:::: The number of committed cores available to the node.\n\n*memory*:::: The amount of memory available to the node. This value\nshould include unit (eg. 3072MB or 9GB). If no units are specified,\n*MB is assumed*.\n\n*localSSD*:::: Optional. The amount of SSD space available on the\nnode. This value should include unit (eg. 3072MB or 9GB). If no\nunits are specified, *GB is assumed*. If this key is not specified, local SSD is\nunconstrained.\n      ")