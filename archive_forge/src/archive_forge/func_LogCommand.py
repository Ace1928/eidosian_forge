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
def LogCommand(prog, args):
    """Log (to debug) the command/arguments being run in a standard format.

  `gcloud feedback` depends on this format.

  Example format is:

      Running [gcloud.example.command] with arguments: [--bar: "baz"]

  Args:
    prog: string, the dotted name of the command being run (ex.
        "gcloud.foos.list")
    args: argparse.namespace, the parsed arguments from the command line
  """
    specified_args = sorted(six.iteritems(args.GetSpecifiedArgs()))
    arg_string = ', '.join(['{}: "{}"'.format(k, v) for k, v in specified_args])
    log.debug('Running [{}] with arguments: [{}]'.format(prog, arg_string))