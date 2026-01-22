import argparse
import arg_parsers
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import copy
import decimal
import json
import re
from dateutil import tz
from googlecloudsdk.calliope import arg_parsers_usage_text as usage_text
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
from six.moves import zip  # pylint: disable=redefined-builtin
def RegexpValidator(pattern, description):
    """Returns a function that validates a string against a regular expression.

  For example:

  >>> alphanumeric_type = RegexpValidator(
  ...   r'[a-zA-Z0-9]+',
  ...   'must contain one or more alphanumeric characters')
  >>> parser.add_argument('--foo', type=alphanumeric_type)
  >>> parser.parse_args(['--foo', '?'])
  >>> # SystemExit raised and the error "error: argument foo: Bad value [?]:
  >>> # must contain one or more alphanumeric characters" is displayed

  Args:
    pattern: str, the pattern to compile into a regular expression to check
    description: an error message to show if the argument doesn't match

  Returns:
    function: str -> str, usable as an argparse type
  """

    def Parse(value):
        if not re.match(pattern + '$', value):
            raise ArgumentTypeError('Bad value [{0}]: {1}'.format(value, description))
        return value
    return Parse