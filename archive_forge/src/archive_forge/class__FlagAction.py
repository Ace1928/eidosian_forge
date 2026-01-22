from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
from absl import flags
class _FlagAction(argparse.Action):
    """Action class for Abseil non-boolean flags."""

    def __init__(self, option_strings, dest, help, metavar, flag_instance, default=argparse.SUPPRESS):
        """Initializes _FlagAction.

    Args:
      option_strings: See argparse.Action.
      dest: Ignored. The flag is always defined with dest=argparse.SUPPRESS.
      help: See argparse.Action.
      metavar: See argparse.Action.
      flag_instance: absl.flags.Flag, the absl flag instance.
      default: Ignored. The flag always uses dest=argparse.SUPPRESS so it
          doesn't affect the parsing result.
    """
        del dest
        self._flag_instance = flag_instance
        super(_FlagAction, self).__init__(option_strings=option_strings, dest=argparse.SUPPRESS, help=help, metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        """See https://docs.python.org/3/library/argparse.html#action-classes."""
        self._flag_instance.parse(values)
        self._flag_instance.using_default_value = False