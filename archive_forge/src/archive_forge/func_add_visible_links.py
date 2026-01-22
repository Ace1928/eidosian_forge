import html
import os
import re
from os import path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple
from urllib.parse import quote
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.utils import smartquotes
from sphinx import addnodes
from sphinx.builders.html import BuildInfo, StandaloneHTMLBuilder
from sphinx.locale import __
from sphinx.util import logging, status_iterator
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.i18n import format_date
from sphinx.util.osutil import copyfile, ensuredir
def add_visible_links(self, tree: nodes.document, show_urls: str='inline') -> None:
    """Add visible link targets for external links"""

    def make_footnote_ref(doc: nodes.document, label: str) -> nodes.footnote_reference:
        """Create a footnote_reference node with children"""
        footnote_ref = nodes.footnote_reference('[#]_')
        footnote_ref.append(nodes.Text(label))
        doc.note_autofootnote_ref(footnote_ref)
        return footnote_ref

    def make_footnote(doc: nodes.document, label: str, uri: str) -> nodes.footnote:
        """Create a footnote node with children"""
        footnote = nodes.footnote(uri)
        para = nodes.paragraph()
        para.append(nodes.Text(uri))
        footnote.append(para)
        footnote.insert(0, nodes.label('', label))
        doc.note_autofootnote(footnote)
        return footnote

    def footnote_spot(tree: nodes.document) -> Tuple[Element, int]:
        """Find or create a spot to place footnotes.

            The function returns the tuple (parent, index)."""
        fns = list(tree.findall(nodes.footnote))
        if fns:
            fn = fns[-1]
            return (fn.parent, fn.parent.index(fn) + 1)
        for node in tree.findall(nodes.rubric):
            if len(node) == 1 and node.astext() == FOOTNOTES_RUBRIC_NAME:
                return (node.parent, node.parent.index(node) + 1)
        doc = next(tree.findall(nodes.document))
        rub = nodes.rubric()
        rub.append(nodes.Text(FOOTNOTES_RUBRIC_NAME))
        doc.append(rub)
        return (doc, doc.index(rub) + 1)
    if show_urls == 'no':
        return
    if show_urls == 'footnote':
        doc = next(tree.findall(nodes.document))
        fn_spot, fn_idx = footnote_spot(tree)
        nr = 1
    for node in list(tree.findall(nodes.reference)):
        uri = node.get('refuri', '')
        if (uri.startswith('http:') or uri.startswith('https:') or uri.startswith('ftp:')) and uri not in node.astext():
            idx = node.parent.index(node) + 1
            if show_urls == 'inline':
                uri = self.link_target_template % {'uri': uri}
                link = nodes.inline(uri, uri)
                link['classes'].append(self.css_link_target_class)
                node.parent.insert(idx, link)
            elif show_urls == 'footnote':
                label = FOOTNOTE_LABEL_TEMPLATE % nr
                nr += 1
                footnote_ref = make_footnote_ref(doc, label)
                node.parent.insert(idx, footnote_ref)
                footnote = make_footnote(doc, label, uri)
                fn_spot.insert(fn_idx, footnote)
                footnote_ref['refid'] = footnote['ids'][0]
                footnote.add_backref(footnote_ref['ids'][0])
                fn_idx += 1