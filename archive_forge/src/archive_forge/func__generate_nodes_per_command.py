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
def _generate_nodes_per_command(self, title, command_name, command_class, ignored_opts):
    """Generate the relevant Sphinx nodes.

        This doesn't bother using raw docutils nodes as they simply don't offer
        the power of directives, like Sphinx's 'option' directive. Instead, we
        generate reStructuredText and parse this in a nested context (to obtain
        correct header levels). Refer to [1] for more information.

        [1] http://www.sphinx-doc.org/en/stable/extdev/markupapi.html

        :param title: Title of command
        :param command_name: Name of command, as used on the command line
        :param command_class: Subclass of :py:class:`cliff.command.Command`
        :param prefix: Prefix to apply before command, if any
        :param ignored_opts: A list of options to exclude from output, if any
        :returns: A list of nested docutil nodes
        """
    command = command_class(None, None)
    if not getattr(command, 'app_dist_name', None):
        command.app_dist_name = self.env.config.autoprogram_cliff_app_dist_name
    parser = command.get_parser(command_name)
    ignored_opts = ignored_opts or []
    self._drop_ignored_options(parser, ignored_opts)
    section = nodes.section('', nodes.title(text=title), ids=[nodes.make_id(title)], names=[nodes.fully_normalize_name(title)])
    source_name = '<{}>'.format(command.__class__.__name__)
    result = statemachine.ViewList()
    for line in _format_parser(parser):
        result.append(line, source_name)
    self.state.nested_parse(result, 0, section)
    return [section]