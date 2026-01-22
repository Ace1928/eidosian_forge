import base64
import binascii
import re
import string
import six
def IsCommonFalse(value):
    """Checks if the string is a commonly accepted False value.

  Useful if you want most strings to default to True except a few
  accepted values.  This method is case-insensitive.

  Args:
    value: The string to check for true.  Or None.

  Returns:
    True if the string is one of the commonly accepted false values.
    True if value is None.  False otherwise.

  Raises:
    ValueError: when value is something besides a string or None.
  """
    if value is None:
        return True
    if not isinstance(value, str):
        raise ValueError('IsCommonFalse() called with %s type.  Expected string.' % type(value))
    if value:
        return value.strip().lower() in _COMMON_FALSE_STRINGS
    return True