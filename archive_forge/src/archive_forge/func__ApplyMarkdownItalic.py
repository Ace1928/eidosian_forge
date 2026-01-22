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
def _ApplyMarkdownItalic(msg):
    return re.sub('(\\b[a-zA-Z][-a-zA-Z_0-9]*)', base.MARKDOWN_ITALIC + '\\1' + base.MARKDOWN_ITALIC, msg)