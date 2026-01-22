from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
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