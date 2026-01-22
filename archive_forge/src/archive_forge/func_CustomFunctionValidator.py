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
def CustomFunctionValidator(fn, description, parser=None):
    """Returns a function that validates the input by running it through fn.

  For example:

  >>> def isEven(val):
  ...   return val % 2 == 0
  >>> even_number_parser = arg_parsers.CustomFunctionValidator(
        isEven, 'This is not even!', parser=arg_parsers.BoundedInt(0))
  >>> parser.add_argument('--foo', type=even_number_parser)
  >>> parser.parse_args(['--foo', '3'])
  >>> # SystemExit raised and the error "error: argument foo: Bad value [3]:
  >>> # This is not even!" is displayed

  Args:
    fn: str -> boolean
    description: an error message to show if boolean function returns False
    parser: an arg_parser that is applied to to value before validation. The
      value is also returned by this parser.

  Returns:
    function: str -> str, usable as an argparse type
  """

    def Parse(value):
        """Validates and returns a custom object from an argument string value."""
        try:
            parsed_value = parser(value) if parser else value
        except ArgumentTypeError:
            pass
        else:
            if fn(parsed_value):
                return parsed_value
        encoded_value = console_attr.SafeText(value)
        formatted_err = 'Bad value [{0}]: {1}'.format(encoded_value, description)
        raise ArgumentTypeError(formatted_err)
    return Parse