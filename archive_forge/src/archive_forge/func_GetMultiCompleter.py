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
def GetMultiCompleter(individual_completer):
    """Create a completer to handle completion for comma separated lists.

  Args:
    individual_completer: A function that completes an individual element.

  Returns:
    A function that completes the last element of the list.
  """

    def MultiCompleter(prefix, parsed_args, **kwargs):
        start = ''
        lst = prefix.rsplit(',', 1)
        if len(lst) > 1:
            start = lst[0] + ','
            prefix = lst[1]
        matches = individual_completer(prefix, parsed_args, **kwargs)
        return [start + match for match in matches]
    return MultiCompleter