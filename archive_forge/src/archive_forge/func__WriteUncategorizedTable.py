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
def _WriteUncategorizedTable(command, elements, element_type, writer):
    """Helper method to GetUncategorizedUsage().

  The elements are written to a markdown table with a special heading. Element
  names are printed in the first column, and help snippet text is printed in the
  second. No categorization is performed.

  Args:
    command: calliope._CommandCommon, The command object that we're helping.
    elements: an iterable over backend.CommandCommon, The sub-elements that
      we're printing to the table.
    element_type: str, The type of elements we are dealing with. Usually
      'groups' or 'commands'.
    writer: file-like, Receiver of the written output.
  """
    writer.write('# Available {element_type} for {group}:\n'.format(element_type=element_type, group=' '.join(command.GetPath())))
    writer.write('---------------------- | ---\n')
    for element in sorted(elements, key=lambda e: e.name):
        if element.IsHidden():
            continue
        writer.write('{name} | {description}\n'.format(name=element.name.replace('_', '-'), description=element.short_help))