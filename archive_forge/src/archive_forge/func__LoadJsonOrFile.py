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
def _LoadJsonOrFile(self, arg_value):
    """Loads json string or file into a dictionary.

    Args:
      arg_value: str, path to a json or yaml file or json string

    Returns:
      Dictionary [str: str] where the value is a json string or other String
        value
    """
    from googlecloudsdk.core import yaml
    file_path_pattern = '^\\S*\\.(yaml|json)$'
    if re.match(file_path_pattern, arg_value):
        arg_value = FileContents()(arg_value)
    if self._LooksLikeJson(arg_value):
        json_value = yaml.load(arg_value)
    else:
        json_value = arg_value
    return self._StringifyDictValues(json_value)