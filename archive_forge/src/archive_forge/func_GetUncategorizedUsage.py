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
def GetUncategorizedUsage(command):
    """Constructs a Usage markdown string for uncategorized command groups.

  The string is formatted as two tables, one for the subgroups and one for the
  subcommands. Each sub-element is printed in its corresponding table together
  with a short summary describing its functionality.

  Args:
    command: calliope._CommandCommon, the command object that we're helping.

  Returns:
    str, The command Usage markdown string as described above.
  """
    buf = io.StringIO()
    if command.groups:
        _WriteUncategorizedTable(command, command.groups.values(), 'groups', buf)
    if command.commands:
        buf.write('\n')
        _WriteUncategorizedTable(command, command.commands.values(), 'commands', buf)
    return buf.getvalue()