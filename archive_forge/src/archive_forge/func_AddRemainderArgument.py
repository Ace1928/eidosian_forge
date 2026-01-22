from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import argparse
import collections
import io
import itertools
import os
import re
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base  # pylint: disable=unused-import
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import suggest_commands
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
import six
def AddRemainderArgument(self, *args, **kwargs):
    """Add an argument representing '--' followed by anything.

    This argument is bound to the parser, so the parser can use its helper
    methods to parse.

    Args:
      *args: The arguments for the action.
      **kwargs: They keyword arguments for the action.

    Raises:
      ArgumentException: If there already is a Remainder Action bound to this
      parser.

    Returns:
      The created action.
    """
    if self._remainder_action:
        self._Error(parser_errors.ArgumentException('There can only be one pass through argument.'))
    kwargs['action'] = arg_parsers.RemainderAction
    self._remainder_action = self.add_argument(*args, **kwargs)
    return self._remainder_action