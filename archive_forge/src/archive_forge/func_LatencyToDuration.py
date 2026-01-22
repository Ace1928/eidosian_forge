from __future__ import absolute_import
import re
def LatencyToDuration(value):
    """Convert valid pending latency argument to a Duration value of seconds.

  Args:
    value: A string in the form X.Xs or XXms.

  Returns:
    Duration value of the given argument.

  Raises:
    ValueError: if the given value isn't parseable.
  """
    if not re.compile(appinfo._PENDING_LATENCY_REGEX).match(value):
        raise ValueError('Unrecognized latency: %s' % value)
    if value == 'automatic':
        return None
    if value.endswith('ms'):
        return '%ss' % (float(value[:-2]) / _MILLISECONDS_PER_SECOND)
    else:
        return value