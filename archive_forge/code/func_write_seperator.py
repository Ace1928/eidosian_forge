import sys
from docutils import nodes
from docutils.statemachine import ViewList
from docutils.parsers.rst import Directive
from sphinx.util.nodes import nested_parse_with_titles
def write_seperator():
    """Write a table separator row"""
    sep = ['=' * c for c in column_lengths]
    return write_row(*sep)