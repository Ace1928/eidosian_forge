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
def _IsPositional(arg):
    """Returns True if arg is a positional or group that contains a positional."""
    if arg.is_hidden:
        return False
    if arg.is_positional:
        return True
    if arg.is_group:
        for a in arg.arguments:
            if _IsPositional(a):
                return True
    return False