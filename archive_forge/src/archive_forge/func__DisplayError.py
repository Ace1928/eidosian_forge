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
def _DisplayError(component_trace):
    """Prints the Fire trace and the error to stdout."""
    result = component_trace.GetResult()
    output = []
    show_help = False
    for help_flag in ('-h', '--help'):
        if help_flag in component_trace.elements[-1].args:
            show_help = True
    if show_help:
        command = '{cmd} -- --help'.format(cmd=component_trace.GetCommand())
        print('INFO: Showing help with the command {cmd}.\n'.format(cmd=pipes.quote(command)), file=sys.stderr)
        help_text = helptext.HelpText(result, trace=component_trace, verbose=component_trace.verbose)
        output.append(help_text)
        Display(output, out=sys.stderr)
    else:
        print(formatting.Error('ERROR: ') + component_trace.elements[-1].ErrorAsStr(), file=sys.stderr)
        error_text = helptext.UsageText(result, trace=component_trace, verbose=component_trace.verbose)
        print(error_text, file=sys.stderr)