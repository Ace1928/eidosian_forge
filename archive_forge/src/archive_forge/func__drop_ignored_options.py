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
def _drop_ignored_options(self, parser, ignored_opts):
    for action in list(parser._actions):
        for option_string in action.option_strings:
            if option_string in ignored_opts:
                del parser._actions[parser._actions.index(action)]
                break