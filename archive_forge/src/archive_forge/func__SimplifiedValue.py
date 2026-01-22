from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
def _SimplifiedValue(validator, value):
    """Convert any value to simplified collections and basic types.

  Args:
    validator: An instance of Validator that corresponds with 'value'.
      May also be 'str' or 'int' if those were used instead of a full
      Validator.
    value: Value to convert to simplified collections.

  Returns:
    The value as a dictionary if it is a ValidatedBase object.  A list of
    items converted to simplified collections if value is a list
    or a tuple. Otherwise, just the value.
  """
    if isinstance(value, ValidatedBase):
        return value.ToDict()
    elif isinstance(value, (list, tuple)):
        return [_SimplifiedValue(validator, item) for item in value]
    elif isinstance(validator, Validator):
        return validator.ToValue(value)
    return value