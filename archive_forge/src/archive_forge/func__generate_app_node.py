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
def _generate_app_node(self, app, application_name):
    ignored_opts = self._get_ignored_opts()
    parser = app.parser
    self._drop_ignored_options(parser, ignored_opts)
    parser.prog = application_name
    source_name = '<{}>'.format(app.__class__.__name__)
    result = statemachine.ViewList()
    for line in _format_parser(parser):
        result.append(line, source_name)
    section = nodes.section()
    self.state.nested_parse(result, 0, section)
    return section.children