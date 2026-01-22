from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import copy
import difflib
import enum
import io
import re
import sys
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import arg_parsers_usage_text
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import util as format_util
import six
def GetSuggestion(self, arg):
    """Find the item that is closest to what was attempted.

    Args:
      arg: str, The argument provided.

    Returns:
      str, The closest match.
    """
    if not self._choices:
        return None
    match = difflib.get_close_matches(arg.lower(), [six.text_type(c) for c in self._choices], 1)
    if match:
        choice = [c for c in self._choices if six.text_type(c) == match[0]][0]
        return self._choices[choice]
    return self._choices[match[0]] if match else None