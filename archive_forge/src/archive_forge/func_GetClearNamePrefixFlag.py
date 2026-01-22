from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
def GetClearNamePrefixFlag():
    """Gets the --clear-name-prefix flag."""
    help_text = '  Clears the name prefix for the system generated reservations.\n  '
    return base.Argument('--clear-name-prefix', action='store_true', help=help_text)