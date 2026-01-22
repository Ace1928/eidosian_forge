from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
def GetClearLocalSsdFlag():
    """Gets the --clear-local-ssd flag."""
    help_text = '  Remove all local ssd information on the future reservation.\n  '
    return base.Argument('--clear-local-ssd', action='store_true', help=help_text)