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
def _format_positional_action(action):
    """Format a positional action."""
    if action.help == argparse.SUPPRESS:
        return
    yield '.. option:: {}'.format((action.metavar or action.dest).strip('<>[]() '))
    if action.help:
        yield ''
        for line in statemachine.string2lines(action.help, tab_width=4, convert_whitespace=True):
            yield _indent(line)