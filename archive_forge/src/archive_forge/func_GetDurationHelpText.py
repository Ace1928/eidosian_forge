from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
def GetDurationHelpText():
    """Gets the --duration help text."""
    help_text = '  Alternate way of specifying time in the number of seconds to terminate\n  capacity request relative to the start time of a request.\n  '
    return help_text