from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
def PrettyTimeDelta(delta):
    """Pretty print the given time delta.

  Rounds down.

  >>> _PrettyTimeDelta(datetime.timedelta(seconds=0))
  '0 seconds'
  >>> _PrettyTimeDelta(datetime.timedelta(minutes=1))
  '1 minute'
  >>> _PrettyTimeDelta(datetime.timedelta(hours=2))
  '2 hours'
  >>> _PrettyTimeDelta(datetime.timedelta(days=3))
  '3 days'

  Args:
    delta: a datetime.timedelta object

  Returns:
    str, a human-readable version of the time delta
  """
    seconds = int(_TotalSeconds(delta))
    num = seconds
    unit = 'second'
    for u, seconds_per in _SECONDS_PER.items():
        if seconds < seconds_per:
            break
        unit = u
        num = seconds // seconds_per
    return '{0} {1}'.format(num, Pluralize(num, unit))