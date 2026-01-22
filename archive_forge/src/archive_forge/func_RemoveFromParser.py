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
def RemoveFromParser(self, parser):
    """Removes this flag from the given parser.

    Args:
      parser: The argparse parser.
    """
    flag = self.__GetFlag(parser)
    if flag:
        name = flag.option_strings[0]
        conflicts = [(name, flag)]
        no_name = '--no-' + name[2:]
        for no_flag in itertools.chain(parser.flag_args, parser.ancestor_flag_args):
            if no_name in no_flag.option_strings:
                conflicts.append((no_name, no_flag))
        flag.container._handle_conflict_resolve(flag, conflicts)
        for _, flag in conflicts:
            parser.defaults.pop(flag.dest, None)
            if flag.dest in parser.dests:
                parser.dests.remove(flag.dest)
            if flag in parser.flag_args:
                parser.flag_args.remove(flag)
            if flag in parser.arguments:
                parser.arguments.remove(flag)