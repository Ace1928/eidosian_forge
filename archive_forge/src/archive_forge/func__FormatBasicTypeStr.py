from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import six
def _FormatBasicTypeStr(arg_type):
    """Returns a user friendly name of a primitive arg_type.

  Args:
    arg_type: type | str | None, expected user input type

  Returns:
    String representation of the type
  """
    if not arg_type:
        return None
    if isinstance(arg_type, str):
        return arg_type
    if arg_type is int:
        return 'int'
    if arg_type is float:
        return 'float'
    if arg_type is bool:
        return 'boolean'
    return 'string'