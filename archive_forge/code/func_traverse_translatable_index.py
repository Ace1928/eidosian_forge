import re
import unicodedata
from typing import (TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Set, Tuple, Type,
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import Directive
from docutils.parsers.rst.states import Inliner
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.locale import __
from sphinx.util import logging
def traverse_translatable_index(doctree: Element) -> Iterable[Tuple[Element, List['IndexEntry']]]:
    """Traverse translatable index node from a document tree."""
    matcher = NodeMatcher(addnodes.index, inline=False)
    for node in doctree.findall(matcher):
        if 'raw_entries' in node:
            entries = node['raw_entries']
        else:
            entries = node['entries']
        yield (node, entries)