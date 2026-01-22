from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
class Preferred(Normalized):
    """A non-deprecated field when there's a deprecated one.

  For use with Deprecated. Only works as a field on Validated.

  Both fields will work for value access. It's an error to set both the
  deprecated and the corresponding preferred field.
  """

    def __init__(self, deprecated, validator, default=None):
        """Initializer for Preferred.

    Args:
      deprecated: The name of the corresponding deprecated field
      validator: The validator for the actual value of this field.
      default: The default value for this field.
    """
        super(Preferred, self).__init__(default=None)
        self.validator = AsValidator(validator)
        self.deprecated = deprecated
        self.synthetic_default = default

    def CheckFieldInitialized(self, value, key, obj):
        deprecated_value = obj.GetUnnormalized(self.deprecated)
        if value is not None and deprecated_value is not None:
            raise ValidationError('Only one of the two fields [{}] (preferred) and [{}] (deprecated) may be set.'.format(key, self.deprecated))
        if deprecated_value is not None:
            return
        if not self.synthetic_default:
            self.validator.CheckFieldInitialized(value, key, obj)

    def Get(self, value, key, obj):
        if value is not None:
            return value
        deprecated_value = obj.GetUnnormalized(self.deprecated)
        if deprecated_value is not None:
            return deprecated_value
        return self.synthetic_default