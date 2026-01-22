from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
def ConvertDurationToSeconds(duration):
    """Convert a string of duration in any form to seconds.

  Args:
    duration: A string of any valid form of duration, such as `10d`, `1w`, `36h`

  Returns:
    A string of duration counted in seconds, such as `1000s`

  Raises:
    BadArgumentExpection: the input duration is mal-formatted.
  """
    try:
        return times.FormatDurationForJson(times.ParseDuration(duration))
    except times.DurationSyntaxError as duration_error:
        raise exceptions.BadArgumentException('--column-families/change-stream-retention-period', str(duration_error))
    except times.DurationValueError as duration_error:
        raise exceptions.BadArgumentException('--column-families/change-stream-retention-period', str(duration_error))