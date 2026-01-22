from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import re
import sys
import textwrap
from googlecloudsdk.calliope import walker
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import module_util
from googlecloudsdk.core.util import files
import six
class FlagOrPositional(Argument):
    """Group, Flag or Positional argument.

  Attributes:
    category: str, The argument help category name.
    completer: str, Resource completer module path.
    default: (self.type), The default flag value or None if no default.
    description: str, The help text.
    name: str, The normalized name ('_' => '-').
    nargs: {0, 1, '?', '*', '+'}
    value: str, The argument value documentation name.
  """

    def __init__(self, arg, name):
        super(FlagOrPositional, self).__init__(arg)
        self.category = getattr(arg, LOOKUP_CATEGORY, '')
        completer = getattr(arg, LOOKUP_COMPLETER, None)
        if completer:
            try:
                completer_class = completer.completer_class
            except AttributeError:
                completer_class = completer
            completer = module_util.GetModulePath(completer_class)
        self.completer = completer
        self.default = arg.default
        self.description = _NormalizeDescription(_GetDescription(arg))
        self.name = six.text_type(name)
        self.nargs = six.text_type(arg.nargs or 0)
        if arg.metavar:
            self.value = six.text_type(arg.metavar)
        else:
            self.value = self.name.lstrip('-').replace('-', '_').upper()
        self._Scrub()

    def _Scrub(self):
        """Scrubs private paths in the default value and description.

    Argument default values and "The default is ..." description text are the
    only places where dynamic private file paths can leak into the cli_tree.
    This method is called on all args.

    The test is rudimentary but effective. Any default value that looks like an
    absolute path on unix or windows is scrubbed. The default value is set to
    None and the trailing "The default ... is ..." sentence in the description,
    if any, is deleted. It's OK to be conservative here and match aggressively.
    """
        if not isinstance(self.default, six.string_types):
            return
        if not re.match('/|[A-Za-z]:\\\\', self.default):
            return
        self.default = None
        match = re.match('(.*\\.) The default (value )?is ', self.description, re.DOTALL)
        if match:
            self.description = match.group(1)