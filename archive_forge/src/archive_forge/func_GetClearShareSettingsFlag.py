from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
def GetClearShareSettingsFlag():
    """Gets the --clear-share-settings help text."""
    help_text = '  Clear share settings on future reservation. This will result in non-shared\n  future reservation.\n  '
    return base.Argument('--clear-share-settings', action='store_true', help=help_text)