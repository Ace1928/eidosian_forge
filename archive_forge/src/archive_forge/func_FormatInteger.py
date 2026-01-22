from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import six
def FormatInteger(value, type_abbr='B'):
    """Returns a pretty string representation of an ISO Decimal value.

  Args:
    value: A scaled integer value.
    type_abbr: The optional type abbreviation suffix, validated but otherwise
      ignored.

  Returns:
    The formatted scaled integer value.
  """
    for suffix, size in reversed(sorted(six.iteritems(_ISO_IEC_UNITS), key=lambda value: (value[1], value[0]))):
        if size <= value and (not value % size):
            return '{}{}{}'.format(value // size, suffix, type_abbr)
    return '{}{}'.format(value, type_abbr)