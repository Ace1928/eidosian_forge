from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import datetime
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
def _FormatFromParts():
    """Returns the formatted time from broken down time parts in r.

    Raises:
      TypeError: For invalid time part errors.
      ValueError: For time conversion errors or not enough valid time parts.

    Returns:
      The formatted time from broken down time parts in r.
    """
    valid = 0
    parts = []
    now = datetime.datetime.now(tz_in)
    for part in ('year', 'month', 'day', 'hour', 'minute', 'second'):
        value = resource_property.Get(r, [part], None)
        if value is None:
            value = getattr(now, part, 0)
        else:
            valid += 1
        parts.append(int(value))
    parts.append(0)
    for i, part in enumerate(['nanosecond', 'microsecond', 'millisecond']):
        value = resource_property.Get(r, [part], None)
        if value is not None:
            parts[-1] += int(int(value) * 1000 ** (i - 1))
    if valid < 3:
        raise ValueError
    parts.append(tz_in)
    dt = datetime.datetime(*parts)
    return times.FormatDateTime(dt, format, tz_out)