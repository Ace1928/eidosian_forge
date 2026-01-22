from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from typing import Any
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddInstanceFlexibilityPolicyArgs(parser: Any, is_update: bool=False) -> None:
    """Adds instance flexibility policy args."""
    parser.add_argument('--instance-selection-machine-types', type=arg_parsers.ArgList(), metavar='MACHINE_TYPE', help='Primary machine types to use for the Compute Engine instances that will be created with the managed instance group. If not provided, machine type specified in the instance template will be used.')
    parser.add_argument('--instance-selection', help='Named selection of machine types with an optional rank. eg. --instance-selection="name=instance-selection-1,machine-type=e2-standard-8,machine-type=t2d-standard-8,rank=0"', metavar='name=NAME,machine-type=MACHINE_TYPE[,machine-type=MACHINE_TYPE...][,rank=RANK]', type=ArgMultiValueDict(), action=arg_parsers.FlattenAction())
    if is_update:
        parser.add_argument('--remove-instance-selections-all', action='store_true', hidden=True, help='Remove all instance selections from the instance flexibility. policy.')
        parser.add_argument('--remove-instance-selections', type=arg_parsers.ArgList(), metavar='INSTANCE_SELECTION_NAME', hidden=True, help='Remove instance selections from the instance flexibility policy.')