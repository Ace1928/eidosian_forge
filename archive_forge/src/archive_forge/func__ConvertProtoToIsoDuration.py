from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.core.util import times
def _ConvertProtoToIsoDuration(proto_duration_str):
    """Convert a given 'proto duration' string to an ISO8601 duration string."""
    return times.FormatDuration(times.ParseDuration(proto_duration_str, True))