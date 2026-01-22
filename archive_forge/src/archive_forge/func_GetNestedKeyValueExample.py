from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import six
def GetNestedKeyValueExample(key_type, value_type, shorthand):
    """Formats example key-value input for flag of arg_type.

  If key_type and value_type are callable types str, returns

    string=string (shorthand) or
    "string": "string" (non-shorthand)

  If key_type is a static string value such as x, returns

    x=string (shorthand) or
    "x": "string" (non-shorthand).

  If key_type or value_type are None, returns string representation of
  key or value

  Args:
    key_type: Callable[[str], Any] | str | None, type function for the key
    value_type: Callable[[str], Any] | None, type function for the value
    shorthand: bool, whether to display the example in shorthand

  Returns:
    str, example of key-value pair
  """
    key_str = _GetNestedValueExample(key_type, shorthand)
    value_str = _GetNestedValueExample(value_type, shorthand)
    if IsHidden(key_type) or IsHidden(value_type):
        return None
    elif not key_str or not value_str:
        return key_str or value_str
    elif shorthand:
        return '{}={}'.format(key_str, value_str)
    else:
        return '{}: {}'.format(key_str, value_str)