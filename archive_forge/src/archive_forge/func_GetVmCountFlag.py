from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
def GetVmCountFlag(required=True):
    help_text = '  The number of VM instances that are allocated to this reservation.\n  The value of this field must be an int in the range [1, 1000].\n  '
    return base.Argument('--vm-count', required=required, type=int, help=help_text)