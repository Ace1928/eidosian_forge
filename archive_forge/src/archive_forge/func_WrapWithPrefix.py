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
def WrapWithPrefix(prefix, message, indent, length, spacing, writer=sys.stdout):
    """Helper function that does two-column writing.

  If the first column is too long, the second column begins on the next line.

  Args:
    prefix: str, Text for the first column.
    message: str, Text for the second column.
    indent: int, Width of the first column.
    length: int, Width of both columns, added together.
    spacing: str, Space to put on the front of prefix.
    writer: file-like, Receiver of the written output.
  """

    def W(s):
        writer.write(s)

    def Wln(s):
        W(s + '\n')
    message = ('\n%%%ds' % indent % ' ').join(textwrap.TextWrapper(break_on_hyphens=False, width=length - indent).wrap(message.replace(' | ', '&| '))).replace('&|', ' |')
    if len(prefix) > indent - len(spacing) - 2:
        Wln('%s%s' % (spacing, prefix))
        W('%%%ds' % indent % ' ')
        Wln(message)
    else:
        W('%s%s' % (spacing, prefix))
        Wln('%%%ds %%s' % (indent - len(prefix) - len(spacing) - 1) % (' ', message))