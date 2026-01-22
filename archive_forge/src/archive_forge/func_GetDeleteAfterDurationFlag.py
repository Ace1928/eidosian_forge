from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
def GetDeleteAfterDurationFlag():
    help_text = '  Automatically deletes the reservations after a specified number of\n  days, hours, minutes, or seconds from its creation. For example,\n  specify 30m for 30 minutes, or 1d2h3m4s for 1 day, 2 hours,\n  3 minutes, and 4 seconds. For more information, see $ gcloud topic datetimes.\n  '
    return base.Argument('--delete-after-duration', type=arg_parsers.Duration(), help=help_text)