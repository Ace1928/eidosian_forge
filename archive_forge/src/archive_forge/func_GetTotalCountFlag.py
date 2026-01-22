from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
def GetTotalCountFlag(required=True):
    """Gets the --total-count flag."""
    help_text = '  The total number of instances for which capacity assurance is requested at a\n  future time period.\n  '
    return base.Argument('--total-count', required=required, type=int, help=help_text)