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