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
def InvalidInputErrorMessage(unit_scales):
    """Constructs an error message for exception thrown invalid input.

  Args:
    unit_scales: list, A list of strings with units that will be recommended to
      user.

  Returns:
    str: The message to use for the exception.
  """
    return 'given value must be of the form DECIMAL[UNITS] where units can be one of {0} and value must be a whole number of Bytes'.format(', '.join(unit_scales))