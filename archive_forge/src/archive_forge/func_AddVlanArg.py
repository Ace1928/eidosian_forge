from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddVlanArg(parser):
    parser.add_argument('--vlan', type=int, help='\n        VLAN tag of a VLAN based network interface, must be in range from 2 to\n        4094 inclusively. This field is mandatory if the parent network\n        interface name is set.\n      ')