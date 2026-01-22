import argparse
import fnmatch
import importlib
import inspect
import re
import sys
from docutils import nodes
from docutils.parsers import rst
from docutils.parsers.rst import directives
from docutils import statemachine
from cliff import app
from cliff import commandmanager
def _format_optional_action(action):
    """Format an optional action."""
    if action.help == argparse.SUPPRESS:
        return
    if action.nargs == 0:
        yield '.. option:: {}'.format(', '.join(action.option_strings))
    else:
        option_strings = [' '.join([x, action.metavar or '<{}>'.format(action.dest.upper())]) for x in action.option_strings]
        yield '.. option:: {}'.format(', '.join(option_strings))
    if action.help:
        yield ''
        for line in statemachine.string2lines(action.help, tab_width=4, convert_whitespace=True):
            yield _indent(line)