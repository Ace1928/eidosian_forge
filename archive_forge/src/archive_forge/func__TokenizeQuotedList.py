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
def _TokenizeQuotedList(arg_value, delim=',', includes_json=False):
    """Tokenize an argument into a list.

  Deliminators that are inside json will not be split. Even when the
  json is nested, we will not split on the delimitor until we reach the
  json's closing bracket. For example:

    '{"a": [1, 2], "b": 3},{"c": 4}'

  with default delim (',') will be split only on the `,` separating the 2
  json strings i.e.

    [
        '{"a": [1, 2], "b": 3}',
        '{"c": 4}'
    ]

  This also works for strings that contain dictionary pattern. For example:

    'key1={"a": [1, 2], "b": 3},key2={"c": 4}'

  with default delim (',') will be split on the delim (',') separating the
  two strings into

    [
        'key1={"a": [1, 2], "b": 3}',
        'key2={"c": 4}'
    ]


  Args:
    arg_value: str, The raw argument.
    delim: str, The delimiter on which to split the argument string.
    includes_json: str, determines whether to ignore delimiter inside json

  Returns:
    [str], The tokenized list.
  """
    if not arg_value:
        return []
    str_list = _SplitOnDelim(arg_value, delim)
    if not includes_json or delim != ',':
        return str_list
    return _RejoinJsonStrs(str_list, delim, arg_value)