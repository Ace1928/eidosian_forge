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
def Duration(default_unit='s', lower_bound='0', upper_bound=None, parsed_unit='s'):
    """Returns a function that can parse time durations.

  See times.ParseDuration() for details. If the unit is omitted, seconds is
  assumed. The parsed unit is assumed to be seconds, but can be specified as
  ms or us.
  For example:

    parser = Duration()
    assert parser('10s') == 10
    parser = Duration(parsed_unit='ms')
    assert parser('10s') == 10000
    parser = Duration(parsed_unit='us')
    assert parser('10s') == 10000000

  Args:
    default_unit: str, The default duration unit.
    lower_bound: str, An inclusive lower bound for values.
    upper_bound: str, An inclusive upper bound for values.
    parsed_unit: str, The unit that the result should be returned as. Can be
      's', 'ms', or 'us'.

  Raises:
    ArgumentTypeError: If either the lower_bound or upper_bound
      cannot be parsed. The returned function will also raise this
      error if it cannot parse its input. This exception is also
      raised if the returned function receives an out-of-bounds
      input.

  Returns:
    A function that accepts a single time duration as input to be
      parsed an returns an integer if the parsed value is not a fraction;
      Otherwise, a float value rounded up to 4 decimals places.
  """

    def Parse(value):
        """Parses a duration from value and returns it in the parsed_unit."""
        if parsed_unit == 'ms':
            multiplier = 1000
        elif parsed_unit == 'us':
            multiplier = 1000000
        elif parsed_unit == 's':
            multiplier = 1
        else:
            raise ArgumentTypeError(_GenerateErrorMessage('parsed_unit must be one of s, ms, us.'))
        try:
            duration = times.ParseDuration(value, default_suffix=default_unit)
            parsed_value = duration.total_seconds * multiplier
            parsed_int_value = int(parsed_value)
            parsed_rounded_value = round(parsed_value, 4)
            fraction = parsed_rounded_value - parsed_int_value
            if fraction:
                return parsed_rounded_value
            return parsed_int_value
        except times.Error as e:
            message = six.text_type(e).rstrip('.')
            raise ArgumentTypeError(_GenerateErrorMessage('Failed to parse duration: {0}'.format(message, user_input=value)))
    parsed_lower_bound = Parse(lower_bound)
    if upper_bound is None:
        parsed_upper_bound = None
    else:
        parsed_upper_bound = Parse(upper_bound)

    def ParseWithBoundsChecking(value):
        """Same as Parse except bound checking is performed."""
        if value is None:
            return None
        parsed_value = Parse(value)
        if parsed_lower_bound is not None and parsed_value < parsed_lower_bound:
            raise ArgumentTypeError(_GenerateErrorMessage('value must be greater than or equal to {0}'.format(lower_bound), user_input=value))
        if parsed_upper_bound is not None and parsed_value > parsed_upper_bound:
            raise ArgumentTypeError(_GenerateErrorMessage('value must be less than or equal to {0}'.format(upper_bound), user_input=value))
        return parsed_value
    return ParseWithBoundsChecking