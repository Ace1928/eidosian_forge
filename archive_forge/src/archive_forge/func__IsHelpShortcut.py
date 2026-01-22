from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import json
import os
import pipes
import re
import shlex
import sys
import types
from fire import completion
from fire import decorators
from fire import formatting
from fire import helptext
from fire import inspectutils
from fire import interact
from fire import parser
from fire import trace
from fire import value_types
from fire.console import console_io
import six
def _IsHelpShortcut(component_trace, remaining_args):
    """Determines if the user is trying to access help without '--' separator.

  For example, mycmd.py --help instead of mycmd.py -- --help.

  Args:
    component_trace: (FireTrace) The trace for the Fire command.
    remaining_args: List of remaining args that haven't been consumed yet.
  Returns:
    True if help is requested, False otherwise.
  """
    show_help = False
    if remaining_args:
        target = remaining_args[0]
        if target in ('-h', '--help'):
            component = component_trace.GetResult()
            if inspect.isclass(component) or inspect.isroutine(component):
                fn_spec = inspectutils.GetFullArgSpec(component)
                _, remaining_kwargs, _ = _ParseKeywordArgs(remaining_args, fn_spec)
                show_help = target in remaining_kwargs
            else:
                members = dict(inspect.getmembers(component))
                show_help = target not in members
    if show_help:
        component_trace.show_help = True
        command = '{cmd} -- --help'.format(cmd=component_trace.GetCommand())
        print('INFO: Showing help with the command {cmd}.\n'.format(cmd=pipes.quote(command)), file=sys.stderr)
    return show_help