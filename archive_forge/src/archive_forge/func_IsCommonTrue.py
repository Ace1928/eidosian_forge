import base64
import binascii
import re
import string
import six
def IsCommonTrue(value):
    """Checks if the string is a commonly accepted True value.

  Useful if you want most strings to default to False except a few
  accepted values.  This method is case-insensitive.

  Args:
    value: The string to check for true.  Or None.

  Returns:
    True if the string is one of the commonly accepted true values.
    False if value is None.  False otherwise.

  Raises:
    ValueError: when value is something besides a string or None.
  """
    if value is None:
        return False
    if not isinstance(value, str):
        raise ValueError('IsCommonTrue() called with %s type.  Expected string.' % type(value))
    if value:
        return value.strip().lower() in _COMMON_TRUE_STRINGS
    return False