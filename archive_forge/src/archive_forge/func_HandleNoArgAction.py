import argparse
import arg_parsers
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import copy
import decimal
import json
import re
from dateutil import tz
from googlecloudsdk.calliope import arg_parsers_usage_text as usage_text
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
from six.moves import zip  # pylint: disable=redefined-builtin
def HandleNoArgAction(none_arg, deprecation_message):
    """Creates an argparse.Action that warns when called with no arguments.

  This function creates an argparse action which can be used to gracefully
  deprecate a flag using nargs=?. When a flag is created with this action, it
  simply log.warning()s the given deprecation_message and then sets the value of
  the none_arg to True.

  This means if you use the none_arg no_foo and attach this action to foo,
  `--foo` (no argument), it will have the same effect as `--no-foo`.

  Args:
    none_arg: a boolean argument to write to. For --no-foo use "no_foo"
    deprecation_message: msg to tell user to stop using with no arguments.

  Returns:
    An argparse action.

  """

    def HandleNoArgActionInit(**kwargs):
        return _HandleNoArgAction(none_arg, deprecation_message, **kwargs)
    return HandleNoArgActionInit