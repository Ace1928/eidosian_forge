from __future__ import annotations
import html
import os
from os import path
from typing import Any
from docutils import nodes
from docutils.nodes import Element, Node, document
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.config import Config
from sphinx.environment.adapters.indexentries import IndexEntries
from sphinx.locale import get_translation
from sphinx.util import logging
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.nodes import NodeMatcher
from sphinx.util.osutil import make_filename_from_project, relpath
from sphinx.util.template import SphinxRenderer
class ToCTreeVisitor(nodes.NodeVisitor):

    def __init__(self, document: document) -> None:
        super().__init__(document)
        self.body: list[str] = []
        self.depth = 0

    def append(self, text: str) -> None:
        self.body.append(text)

    def astext(self) -> str:
        return '\n'.join(self.body)

    def unknown_visit(self, node: Node) -> None:
        pass

    def unknown_departure(self, node: Node) -> None:
        pass

    def visit_bullet_list(self, node: Element) -> None:
        if self.depth > 0:
            self.append('<UL>')
        self.depth += 1

    def depart_bullet_list(self, node: Element) -> None:
        self.depth -= 1
        if self.depth > 0:
            self.append('</UL>')

    def visit_list_item(self, node: Element) -> None:
        self.append('<LI> <OBJECT type="text/sitemap">')
        self.depth += 1

    def depart_list_item(self, node: Element) -> None:
        self.depth -= 1

    def visit_reference(self, node: Element) -> None:
        title = chm_htmlescape(node.astext(), True)
        self.append('    <param name="Name" value="%s">' % title)
        self.append('    <param name="Local" value="%s">' % node['refuri'])
        self.append('</OBJECT>')
        raise nodes.SkipNode