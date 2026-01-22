from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
def GetRequireSpecificReservationFlag():
    """--require-specific-reservation flag."""
    help_text = '  Indicate whether the auto-created reservations can be consumed by VMs with\n  "any reservation" defined. If enabled, then only VMs that target the\n  auto-created reservation by name using `--reservation-affinity=specific` can\n  consume from this reservation. Auto-created reservations delivered with this\n  flag enabled will inherit the name of the future reservation.\n  '
    return base.Argument('--require-specific-reservation', action=arg_parsers.StoreTrueFalseAction, help=help_text)