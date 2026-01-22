from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.tasks import constants
import six
def ConvertStringToCamelCase(string):
    """Takes a 'snake_case' string and converts it to 'camelCase'.

  Args:
    string: The string we want to convert.

  Returns:
    The converted string. Some examples are below:
      min_backoff => minBackoff
      max_retry_duration => maxRetryDuration
  """
    if not hasattr(ConvertStringToCamelCase, 'processed_strings'):
        ConvertStringToCamelCase.processed_strings = {}
    if string in ConvertStringToCamelCase.processed_strings:
        return ConvertStringToCamelCase.processed_strings[string]
    attributes = string.split('_')
    for index, attribute in enumerate(attributes):
        if index == 0:
            continue
        attributes[index] = attribute.capitalize()
    camel_case_string = ''.join(attributes)
    ConvertStringToCamelCase.processed_strings[string] = camel_case_string
    return camel_case_string