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
def _Map(self, arg_value, callback):
    """Applies callback for arg_value.

    Arg_value can be a dictionary, list, or other value.

    Args:
      arg_value: can be a dictionary, list, or other value,
      callback: (key, val) -> key, val, function that accepts key and value
        and returns transformed values.

    Returns:
      dictionary, list, or value with callback operation performed on it.
    """
    if isinstance(arg_value, list) and self.repeated:
        arg_list = []
        for value in arg_value:
            value = self._Map(value, callback)
            arg_list.append(value)
        return arg_list
    if isinstance(arg_value, dict) and self._keyed_values:
        arg_dict = collections.OrderedDict()
        for key, value in arg_value.items():
            key, value = callback(key, value)
            arg_dict[key] = value
        return arg_dict
    _, value = callback(None, arg_value)
    return value