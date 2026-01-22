import optparse
import re
from typing import Callable, Dict
from . import errors
from . import registry as _mod_registry
from . import revisionspec
class ListOption(Option):
    """Option used to provide a list of values.

    On the command line, arguments are specified by a repeated use of the
    option. '-' is a special argument that resets the list. For example,
      --foo=a --foo=b
    sets the value of the 'foo' option to ['a', 'b'], and
      --foo=a --foo=b --foo=- --foo=c
    sets the value of the 'foo' option to ['c'].
    """

    def add_option(self, parser, short_name):
        """Add this option to an Optparse parser."""
        option_strings = ['--%s' % self.name]
        if short_name is not None:
            option_strings.append('-%s' % short_name)
        parser.add_option(*option_strings, action='callback', callback=self._optparse_callback, type='string', metavar=self.argname.upper(), help=self.help, dest=self._param_name, default=[])

    def _optparse_callback(self, option, opt, value, parser):
        values = getattr(parser.values, self._param_name)
        if value == '-':
            del values[:]
        else:
            values.append(self.type(value))
        if self.custom_callback is not None:
            self.custom_callback(option, self._param_name, values, parser)