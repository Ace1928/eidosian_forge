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
def GetBinarySizePerUnit(suffix, type_abbr='B'):
    """Returns the binary size per unit for binary suffix string.

  Args:
    suffix: str, A case insensitive unit suffix string with optional type
      abbreviation.
    type_abbr: str, The optional case insensitive type abbreviation following
      the suffix.

  Raises:
    ValueError for unknown units.

  Returns:
    The binary size per unit for a unit+type_abbr suffix.
  """
    unit = _DeleteTypeAbbr(suffix.upper(), type_abbr)
    return _BINARY_SIZE_SCALES.get(unit)