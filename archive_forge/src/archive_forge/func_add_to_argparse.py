from __future__ import unicode_literals
import collections
import contextlib
import inspect
import logging
import pprint
import sys
import textwrap
import six
def add_to_argparse(self, optgroup):
    """Add the config variable as an argument to the command line parser."""
    if self.name == 'additional_commands':
        return
    if isinstance(self.default_value, bool):
        optgroup.add_argument('--' + self.name.replace('_', '-'), nargs='?', default=None, const=not self.default_value, type=parse_bool, help=self.helptext)
    elif isinstance(self.default_value, VALUE_TYPES):
        optgroup.add_argument('--' + self.name.replace('_', '-'), type=type(self.default_value), help=self.helptext, choices=self.choices)
    elif self.default_value is None:
        optgroup.add_argument('--' + self.name.replace('_', '-'), help=self.helptext, choices=self.choices)
    elif isinstance(self.default_value, (list, tuple)):
        typearg = None
        if self.default_value:
            typearg = type(self.default_value[0])
        optgroup.add_argument('--' + self.name.replace('_', '-'), nargs='*', type=typearg, help=self.helptext)