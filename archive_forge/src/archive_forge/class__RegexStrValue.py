from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
class _RegexStrValue(object):
    """Simulates the regex object to support recompilation when necessary.

  Used by the RegexStr class to dynamically build and recompile regular
  expression attributes of a validated object.  This object replaces the normal
  object returned from re.compile which is immutable.

  When the value of this object is a string, that string is simply used as the
  regular expression when recompilation is needed.  If the state of this object
  is a list of strings, the strings are joined in to a single 'or' expression.
  """

    def __init__(self, attribute, value, key):
        """Initialize recompilable regex value.

    Args:
      attribute: Attribute validator associated with this regex value.
      value: Initial underlying python value for regex string.  Either a single
        regex string or a list of regex strings.
      key: Name of the field.
    """
        self.__attribute = attribute
        self.__value = value
        self.__regex = None
        self.__key = key

    def __AsString(self, value):
        """Convert a value to appropriate string.

    Returns:
      String version of value with all carriage returns and line feeds removed.
    """
        if issubclass(self.__attribute.expected_type, str):
            cast_value = TYPE_STR(value)
        else:
            cast_value = TYPE_UNICODE(value)
        cast_value = cast_value.replace('\n', '')
        cast_value = cast_value.replace('\r', '')
        return cast_value

    def __BuildRegex(self):
        """Build regex string from state.

    Returns:
      String version of regular expression.  Sequence objects are constructed
      as larger regular expression where each regex in the list is joined with
      all the others as single 'or' expression.
    """
        if isinstance(self.__value, list):
            value_list = self.__value
            sequence = True
        else:
            value_list = [self.__value]
            sequence = False
        regex_list = []
        for item in value_list:
            regex_list.append(self.__AsString(item))
        if sequence:
            return '|'.join(('%s' % item for item in regex_list))
        else:
            return regex_list[0]

    def __Compile(self):
        """Build regular expression object from state.

    Returns:
      Compiled regular expression based on internal value.
    """
        regex = self.__BuildRegex()
        try:
            return re.compile(regex)
        except re.error as e:
            raise ValidationError("Value '%s' for %s does not compile: %s" % (regex, self.__key, e), e)

    @property
    def regex(self):
        """Compiled regular expression as described by underlying value."""
        return self.__Compile()

    def match(self, value):
        """Match against internal regular expression.

    Args:
      value: String to match against regular expression.

    Returns:
      Regular expression object built from underlying value.
    """
        return re.match(self.__BuildRegex(), value)

    def Validate(self):
        """Ensure that regex string compiles."""
        self.__Compile()

    def __str__(self):
        """Regular expression string as described by underlying value."""
        return self.__BuildRegex()

    def __eq__(self, other):
        """Comparison against other regular expression string values."""
        if isinstance(other, _RegexStrValue):
            return self.__BuildRegex() == other.__BuildRegex()
        return str(self) == other

    def __ne__(self, other):
        """Inequality operator for regular expression string value."""
        return not self.__eq__(other)