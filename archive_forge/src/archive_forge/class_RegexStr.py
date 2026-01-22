from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
class RegexStr(Validator):
    """Validates that a string can compile as a regex without errors.

  Use this validator when the value of a field should be a regex.  That
  means that the value must be a string that can be compiled by re.compile().
  The attribute will then be a compiled re object.
  """

    def __init__(self, string_type=six_subset.text_type, default=None):
        """Initialized regex validator.

    Args:
      string_type: Type to be considered a string.
      default: Default value.

    Raises:
      AttributeDefinitionError: if string_type is not a kind of string.
    """
        if default is not None:
            default = _RegexStrValue(self, default, None)
            re.compile(str(default))
        super(RegexStr, self).__init__(default)
        if not issubclass(string_type, six_subset.string_types) or six_subset.is_basestring(string_type):
            raise AttributeDefinitionError('RegexStr fields must be a string type not %s.' % str(string_type))
        self.expected_type = string_type

    def Validate(self, value, key):
        """Validates that the string compiles as a regular expression.

    Because the regular expression might have been expressed as a multiline
    string, this function also strips newlines out of value.

    Args:
      value: String to compile as a regular expression.
      key: Name of the field being validated.

    Raises:
      ValueError when value does not compile as a regular expression.  TypeError
      when value does not match provided string type.
    """
        if isinstance(value, _RegexStrValue):
            return value
        value = _RegexStrValue(self, value, key)
        value.Validate()
        return value

    def ToValue(self, value):
        """Returns the RE pattern for this validator."""
        return str(value)