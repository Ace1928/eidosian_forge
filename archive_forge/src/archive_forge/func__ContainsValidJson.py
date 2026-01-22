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
def _ContainsValidJson(str_value):
    """Checks whether the string contains balanced json."""
    closing_brackets = {'}': '{', ']': '['}
    opening_brackets = set(closing_brackets.values())
    current_brackets = []
    for i in range(len(str_value)):
        if i > 0 and str_value[i - 1] == '\\':
            continue
        ch = str_value[i]
        if ch in closing_brackets:
            matching_brace = closing_brackets[ch]
            if not current_brackets or current_brackets[-1] != matching_brace:
                return False
            current_brackets.pop()
        elif ch in opening_brackets:
            current_brackets.append(ch)
    return not current_brackets