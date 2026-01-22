from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.images import flags
def _ResolveTime(absolute, relative_sec, current_time):
    """Get the RFC 3339 time string for a provided absolute or relative time."""
    if absolute:
        return absolute
    elif relative_sec:
        return (current_time + datetime.timedelta(seconds=relative_sec)).replace(microsecond=0).isoformat()
    else:
        return None