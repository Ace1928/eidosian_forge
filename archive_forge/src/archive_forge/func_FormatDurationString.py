from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def FormatDurationString(duration):
    """Returns the duration string.

  Args:
    duration: the duration, an int. The unit is seconds.

  Returns:
    a duration with string format.
  """
    return '{}s'.format(duration)