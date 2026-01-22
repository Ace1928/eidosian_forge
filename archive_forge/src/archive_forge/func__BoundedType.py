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
def _BoundedType(type_builder, type_description, lower_bound=None, upper_bound=None, unlimited=False):
    """Returns a function that can parse given type within some bound.

  Args:
    type_builder: A callable for building the requested type from the value
      string.
    type_description: str, Description of the requested type (for verbose
      messages).
    lower_bound: of type compatible with type_builder, The value must be >=
      lower_bound.
    upper_bound: of type compatible with type_builder, The value must be <=
      upper_bound.
    unlimited: bool, If True then a value of 'unlimited' means no limit.

  Returns:
    A function that can parse given type within some bound.
  """

    def Parse(value):
        """Parses value as a type constructed by type_builder.

    Args:
      value: str, Value to be converted to the requested type.

    Raises:
      ArgumentTypeError: If the provided value is out of bounds or unparsable.

    Returns:
      Value converted to the requested type.
    """
        if unlimited and value == 'unlimited':
            return None
        try:
            v = type_builder(value)
        except ValueError:
            raise ArgumentTypeError(_GenerateErrorMessage('Value must be {0}'.format(type_description), user_input=value))
        if lower_bound is not None and v < lower_bound:
            raise ArgumentTypeError(_GenerateErrorMessage('Value must be greater than or equal to {0}'.format(lower_bound), user_input=value))
        if upper_bound is not None and upper_bound < v:
            raise ArgumentTypeError(_GenerateErrorMessage('Value must be less than or equal to {0}'.format(upper_bound), user_input=value))
        return v
    return Parse