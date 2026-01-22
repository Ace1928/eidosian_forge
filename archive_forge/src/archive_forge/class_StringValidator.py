from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
class StringValidator(Validator):
    """Verifies property is a valid text string.

  In python 2: inherits from basestring
  In python 3: inherits from str
  """

    def Validate(self, value, key='???'):
        if not isinstance(value, six_subset.string_types):
            raise ValidationError('Value %r for %s is not a valid text string.' % (value, key))
        return value