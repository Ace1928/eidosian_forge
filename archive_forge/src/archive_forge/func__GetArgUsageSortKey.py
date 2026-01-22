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
def _GetArgUsageSortKey(name):
    """Arg name usage string key function for sorted."""
    if not name:
        return (0, '')
    elif name.startswith('--no-'):
        return (3, name[5:], 'x')
    elif name.startswith('--'):
        return (3, name[2:])
    elif name.startswith('-'):
        return (4, name[1:])
    elif name[0].isalpha():
        return (1, '')
    else:
        return (5, name)