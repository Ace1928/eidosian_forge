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
@staticmethod
def ParseUtcTime(s):
    """Parses a string representing a time in UTC into a Datetime object."""
    if not s:
        return None
    try:
        return times.ParseDateTime(s, tzinfo=tz.tzutc())
    except times.Error as e:
        raise ArgumentTypeError(_GenerateErrorMessage('Failed to parse UTC time: {0}'.format(six.text_type(e)), user_input=s))