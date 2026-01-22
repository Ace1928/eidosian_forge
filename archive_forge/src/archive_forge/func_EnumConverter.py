from __future__ import absolute_import
import re
def EnumConverter(prefix):
    """Create conversion function which translates from string to enum value.

  Args:
    prefix: Prefix for enum value. Expected to be an upper-cased value.

  Returns:
    A conversion function which translates from string to enum value.

  Raises:
    ValueError: If an invalid prefix (empty, non-upper-cased, etc.) prefix was
    provided.
  """
    if not prefix:
        raise ValueError('A prefix must be provided')
    if prefix != prefix.upper():
        raise ValueError('Upper-cased prefix must be provided')
    if prefix.endswith('_'):
        raise ValueError('Prefix should not contain a trailing underscore: "%s"' % prefix)
    return lambda value: '_'.join([prefix, str(value).upper()])