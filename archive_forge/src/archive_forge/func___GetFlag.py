from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import contextlib
import enum
from functools import wraps  # pylint:disable=g-importing-member
import itertools
import re
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import display
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_printer
import six
def __GetFlag(self, parser):
    """Returns the flag object in parser."""
    for flag in itertools.chain(parser.flag_args, parser.ancestor_flag_args):
        if self.name in flag.option_strings:
            return flag
    return None