from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
from absl import flags
class _BooleanFlagAction(argparse.Action):
    """Action class for Abseil boolean flags."""

    def __init__(self, option_strings, dest, help, metavar, flag_instance, default=argparse.SUPPRESS):
        """Initializes _BooleanFlagAction.

    Args:
      option_strings: See argparse.Action.
      dest: Ignored. The flag is always defined with dest=argparse.SUPPRESS.
      help: See argparse.Action.
      metavar: See argparse.Action.
      flag_instance: absl.flags.Flag, the absl flag instance.
      default: Ignored. The flag always uses dest=argparse.SUPPRESS so it
          doesn't affect the parsing result.
    """
        del dest, default
        self._flag_instance = flag_instance
        flag_names = [self._flag_instance.name]
        if self._flag_instance.short_name:
            flag_names.append(self._flag_instance.short_name)
        self._flag_names = frozenset(flag_names)
        super(_BooleanFlagAction, self).__init__(option_strings=option_strings, dest=argparse.SUPPRESS, nargs=0, help=help, metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        """See https://docs.python.org/3/library/argparse.html#action-classes."""
        if not isinstance(values, list) or values:
            raise ValueError('values must be an empty list.')
        if option_string.startswith('--'):
            option = option_string[2:]
        else:
            option = option_string[1:]
        if option in self._flag_names:
            self._flag_instance.parse('true')
        else:
            if not option.startswith('no') or option[2:] not in self._flag_names:
                raise ValueError('invalid option_string: ' + option_string)
            self._flag_instance.parse('false')
        self._flag_instance.using_default_value = False