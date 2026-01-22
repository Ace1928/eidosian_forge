from __future__ import absolute_import
import re
from ruamel import yaml
from googlecloudsdk.third_party.appengine._internal import six_subset
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