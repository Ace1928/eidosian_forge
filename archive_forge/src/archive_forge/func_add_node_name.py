import re
import textwrap
from os import path
from typing import (TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Pattern, Set,
from docutils import nodes, writers
from docutils.nodes import Element, Node, Text
from sphinx import __display_version__, addnodes
from sphinx.domains import IndexEntry
from sphinx.domains.index import IndexDomain
from sphinx.errors import ExtensionError
from sphinx.locale import _, __, admonitionlabels
from sphinx.util import logging
from sphinx.util.docutils import SphinxTranslator
from sphinx.util.i18n import format_date
from sphinx.writers.latex import collected_footnote
def add_node_name(name: str) -> str:
    node_id = self.escape_id(name)
    nth, suffix = (1, '')
    while node_id + suffix in self.written_ids or node_id + suffix in self.node_names:
        nth += 1
        suffix = '<%s>' % nth
    node_id += suffix
    self.written_ids.add(node_id)
    self.node_names[node_id] = name
    return node_id