from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
import re
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def _PrintArgGroup(self, arg, depth=0, single=False):
    """Prints an arg group definition list at depth."""
    args = sorted(arg.arguments, key=usage_text.GetArgSortKey) if arg.sort_args else arg.arguments
    heading = []
    if arg.help or arg.is_mutex or arg.is_required:
        if arg.help:
            heading.append(arg.help)
        if arg.disable_default_heading:
            pass
        elif len(args) == 1 or args[0].is_required:
            if arg.is_required:
                heading.append('This must be specified.')
        elif arg.is_mutex:
            if arg.is_required:
                heading.append('Exactly one of these must be specified:')
            else:
                heading.append('At most one of these can be specified:')
        elif arg.is_required:
            heading.append('At least one of these must be specified:')
    if not arg.is_hidden and heading:
        self._out('\n{0} {1}\n\n'.format(':' * (depth + _SECOND_LINE_OFFSET), '\n+\n'.join(heading)).replace('\n\n', '\n+\n'))
        heading = None
        depth += 1
    for a in args:
        if a.is_hidden:
            continue
        if a.is_group:
            single = False
            singleton = usage_text.GetSingleton(a)
            if singleton:
                if not a.help:
                    a = singleton
                else:
                    single = True
        if a.is_group:
            self._PrintArgGroup(a, depth=depth, single=single)
        else:
            self._PrintArgDefinition(a, depth=depth, single=single)