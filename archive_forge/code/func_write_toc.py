from __future__ import annotations
import html
import os
import posixpath
import re
from collections.abc import Iterable
from os import path
from typing import Any, cast
from docutils import nodes
from docutils.nodes import Node
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.environment.adapters.indexentries import IndexEntries
from sphinx.locale import get_translation
from sphinx.util import logging
from sphinx.util.nodes import NodeMatcher
from sphinx.util.osutil import canon_path, make_filename
from sphinx.util.template import SphinxRenderer
def write_toc(self, node: Node, indentlevel: int=4) -> list[str]:
    parts: list[str] = []
    if isinstance(node, nodes.list_item) and self.isdocnode(node):
        compact_paragraph = cast(addnodes.compact_paragraph, node[0])
        reference = cast(nodes.reference, compact_paragraph[0])
        link = reference['refuri']
        title = html.escape(reference.astext()).replace('"', '&quot;')
        item = '<section title="%(title)s" ref="%(ref)s">' % {'title': title, 'ref': link}
        parts.append(' ' * 4 * indentlevel + item)
        bullet_list = cast(nodes.bullet_list, node[1])
        list_items = cast(Iterable[nodes.list_item], bullet_list)
        for list_item in list_items:
            parts.extend(self.write_toc(list_item, indentlevel + 1))
        parts.append(' ' * 4 * indentlevel + '</section>')
    elif isinstance(node, nodes.list_item):
        for subnode in node:
            parts.extend(self.write_toc(subnode, indentlevel))
    elif isinstance(node, nodes.reference):
        link = node['refuri']
        title = html.escape(node.astext()).replace('"', '&quot;')
        item = section_template % {'title': title, 'ref': link}
        item = ' ' * 4 * indentlevel + item
        parts.append(item.encode('ascii', 'xmlcharrefreplace').decode())
    elif isinstance(node, nodes.bullet_list):
        for subnode in node:
            parts.extend(self.write_toc(subnode, indentlevel))
    elif isinstance(node, addnodes.compact_paragraph):
        for subnode in node:
            parts.extend(self.write_toc(subnode, indentlevel))
    return parts