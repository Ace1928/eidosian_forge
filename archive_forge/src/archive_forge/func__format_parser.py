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
def _format_parser(parser):
    """Format the output of an argparse 'ArgumentParser' object.

    Given the following parser::

      >>> import argparse
      >>> parser = argparse.ArgumentParser(prog='hello-world',               description='This is my description.',
              epilog='This is my epilog')
      >>> parser.add_argument('name', help='User name', metavar='<name>')
      >>> parser.add_argument('--language', action='store', dest='lang',               help='Greeting language')

    Returns the following::

      This is my description.

      .. program:: hello-world
      .. code:: shell

          hello-world [-h] [--language LANG] <name>

      .. option:: name

          User name

      .. option:: --language LANG

          Greeting language

      .. option:: -h, --help

          Show this help message and exit

      This is my epilog.
    """
    if parser.description:
        for line in _format_description(parser):
            yield line
        yield ''
    yield '.. program:: {}'.format(parser.prog)
    yield '.. code-block:: shell'
    yield ''
    for line in _format_usage(parser):
        yield _indent(line)
    yield ''
    for action in parser._get_optional_actions():
        for line in _format_optional_action(action):
            yield line
        yield ''
    for action in parser._get_positional_actions():
        for line in _format_positional_action(action):
            yield line
        yield ''
    if parser.epilog:
        for line in _format_epilog(parser):
            yield line
        yield ''