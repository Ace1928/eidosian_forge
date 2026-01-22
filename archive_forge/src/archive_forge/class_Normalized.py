from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
class Normalized(Validator):
    """Normalizes a field on lookup, but serializes with the original value.

  Only works with fields on Validated.
  """

    def Validate(self, value, key):
        return self.validator(value, key)

    def Get(self, value, key, obj):
        """Returns the normalized value. Subclasses must override."""
        raise NotImplementedError('Subclasses must override `Get`!')