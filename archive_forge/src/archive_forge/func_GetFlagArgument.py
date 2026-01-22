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
def GetFlagArgument(self, name):
    """Returns the flag argument object for name.

    Args:
      name: The flag name or Namespace destination.

    Raises:
      UnknownDestinationException: If there is no registered flag arg for name.

    Returns:
      The flag argument object for name.
    """
    if name.startswith('--'):
        dest = name[2:].replace('-', '_')
        flag = name
    else:
        dest = name
        flag = '--' + name.replace('_', '-')
    ai = self._GetCommand().ai
    for arg in ai.flag_args + ai.ancestor_flag_args:
        if dest == arg.dest or (arg.option_strings and flag == arg.option_strings[0]):
            return arg
    raise parser_errors.UnknownDestinationException('No registered flag arg for [{}].'.format(name))