from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
class TimeValue(Validator):
    """Validates time values with units, such as 1h or 3.5d."""
    _EXPECTED_SYNTAX = 'must be a non-negative number followed by a time unit, such as 1h or 3.5d'

    def __init__(self):
        super(TimeValue, self).__init__()
        self.expected_type = str

    def Validate(self, value, key):
        """Validate a time value.

    Args:
      value: Value to validate.
      key: Name of the field being validated.

    Raises:
      ValidationError: if value is not a time value with the expected format.
    """
        if not isinstance(value, six_subset.string_types):
            raise ValidationError("Value '%s' for %s is not a string (%s)" % (value, key, TimeValue._EXPECTED_SYNTAX))
        if not value:
            raise ValidationError('Value for %s is empty (%s)' % (key, TimeValue._EXPECTED_SYNTAX))
        if value[-1] not in 'smhd':
            raise ValidationError("Value '%s' for %s must end with a time unit, one of s (seconds), m (minutes), h (hours), or d (days)" % (value, key))
        try:
            t = float(value[:-1])
        except ValueError:
            raise ValidationError("Value '%s' for %s is not a valid time value (%s)" % (value, key, TimeValue._EXPECTED_SYNTAX))
        if t < 0:
            raise ValidationError("Value '%s' for %s is negative (%s)" % (value, key, TimeValue._EXPECTED_SYNTAX))
        return value