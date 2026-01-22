from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddParentNicNameArg(parser):
    parser.add_argument('--parent-nic-name', type=str, help='\n        Name of the parent network interface of a VLAN based network interface.\n        If this field is specified, vlan must be set.\n      ')