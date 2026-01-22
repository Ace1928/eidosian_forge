from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
def GetMachineType(required=True):
    """Gets the --machine-type flag."""
    help_text = '  The type of machine (name only) that has a fixed number of vCPUs and a fixed\n  amount of memory. You can also specify a custom machine type by using the\n  pattern `custom-number_of_CPUs-amount_of_memory`-for example,\n  `custom-32-29440`.\n  '
    return base.Argument('--machine-type', required=required, help=help_text)