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
def GetPositionalUsage(arg, markdown=False):
    """Create the usage help string for a positional arg.

  Args:
    arg: parser_arguments.Argument, The argument object to be displayed.
    markdown: bool, If true add markdowns.

  Returns:
    str, The string representation for printing.
  """
    var = arg.metavar or arg.dest.upper()
    if markdown:
        var = _ApplyMarkdownItalic(var)
    if arg.nargs == '+':
        return '{var} [{var} ...]'.format(var=var)
    elif arg.nargs == '*':
        return '[{var} ...]'.format(var=var)
    elif arg.nargs == argparse.REMAINDER:
        return '[-- {var} ...]'.format(var=var)
    elif arg.nargs == '?':
        return '[{var}]'.format(var=var)
    else:
        return var