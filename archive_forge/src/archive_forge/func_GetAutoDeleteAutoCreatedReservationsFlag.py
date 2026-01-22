from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
def GetAutoDeleteAutoCreatedReservationsFlag(required=False):
    """Gets the --auto-delete-auto-created-reservations flag."""
    help_text = '  If specified, the auto-created reservations for a future reservation\n  are deleted at the end time (default) or at a specified delete time.\n  '
    return base.Argument('--auto-delete-auto-created-reservations', action=arg_parsers.StoreTrueFalseAction, help=help_text, required=required)