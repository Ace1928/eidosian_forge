from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
from ..inlinepatterns import InlineProcessor
from ..treeprocessors import Treeprocessor
from ..postprocessors import Postprocessor
from .. import util
from collections import OrderedDict
import re
import copy
import xml.etree.ElementTree as etree
class FootnoteExtension(Extension):
    """ Footnote Extension. """

    def __init__(self, **kwargs):
        """ Setup configs. """
        self.config = {'PLACE_MARKER': ['///Footnotes Go Here///', 'The text string that marks where the footnotes go'], 'UNIQUE_IDS': [False, 'Avoid name collisions across multiple calls to `reset()`.'], 'BACKLINK_TEXT': ['&#8617;', "The text string that links from the footnote to the reader's place."], 'SUPERSCRIPT_TEXT': ['{}', "The text string that links from the reader's place to the footnote."], 'BACKLINK_TITLE': ['Jump back to footnote %d in the text', 'The text string used for the title HTML attribute of the backlink. %d will be replaced by the footnote number.'], 'SEPARATOR': [':', 'Footnote separator.']}
        ' Default configuration options. '
        super().__init__(**kwargs)
        self.unique_prefix = 0
        self.found_refs: dict[str, int] = {}
        self.used_refs: set[str] = set()
        self.reset()

    def extendMarkdown(self, md):
        """ Add pieces to Markdown. """
        md.registerExtension(self)
        self.parser = md.parser
        self.md = md
        md.parser.blockprocessors.register(FootnoteBlockProcessor(self), 'footnote', 17)
        FOOTNOTE_RE = '\\[\\^([^\\]]*)\\]'
        md.inlinePatterns.register(FootnoteInlineProcessor(FOOTNOTE_RE, self), 'footnote', 175)
        md.treeprocessors.register(FootnoteTreeprocessor(self), 'footnote', 50)
        md.treeprocessors.register(FootnotePostTreeprocessor(self), 'footnote-duplicate', 15)
        md.postprocessors.register(FootnotePostprocessor(self), 'footnote', 25)

    def reset(self) -> None:
        """ Clear footnotes on reset, and prepare for distinct document. """
        self.footnotes: OrderedDict[str, str] = OrderedDict()
        self.unique_prefix += 1
        self.found_refs = {}
        self.used_refs = set()

    def unique_ref(self, reference: str, found: bool=False) -> str:
        """ Get a unique reference if there are duplicates. """
        if not found:
            return reference
        original_ref = reference
        while reference in self.used_refs:
            ref, rest = reference.split(self.get_separator(), 1)
            m = RE_REF_ID.match(ref)
            if m:
                reference = '%s%d%s%s' % (m.group(1), int(m.group(2)) + 1, self.get_separator(), rest)
            else:
                reference = '%s%d%s%s' % (ref, 2, self.get_separator(), rest)
        self.used_refs.add(reference)
        if original_ref in self.found_refs:
            self.found_refs[original_ref] += 1
        else:
            self.found_refs[original_ref] = 1
        return reference

    def findFootnotesPlaceholder(self, root: etree.Element) -> tuple[etree.Element, etree.Element, bool] | None:
        """ Return ElementTree Element that contains Footnote placeholder. """

        def finder(element):
            for child in element:
                if child.text:
                    if child.text.find(self.getConfig('PLACE_MARKER')) > -1:
                        return (child, element, True)
                if child.tail:
                    if child.tail.find(self.getConfig('PLACE_MARKER')) > -1:
                        return (child, element, False)
                child_res = finder(child)
                if child_res is not None:
                    return child_res
            return None
        res = finder(root)
        return res

    def setFootnote(self, id: str, text: str) -> None:
        """ Store a footnote for later retrieval. """
        self.footnotes[id] = text

    def get_separator(self) -> str:
        """ Get the footnote separator. """
        return self.getConfig('SEPARATOR')

    def makeFootnoteId(self, id: str) -> str:
        """ Return footnote link id. """
        if self.getConfig('UNIQUE_IDS'):
            return 'fn%s%d-%s' % (self.get_separator(), self.unique_prefix, id)
        else:
            return 'fn{}{}'.format(self.get_separator(), id)

    def makeFootnoteRefId(self, id: str, found: bool=False) -> str:
        """ Return footnote back-link id. """
        if self.getConfig('UNIQUE_IDS'):
            return self.unique_ref('fnref%s%d-%s' % (self.get_separator(), self.unique_prefix, id), found)
        else:
            return self.unique_ref('fnref{}{}'.format(self.get_separator(), id), found)

    def makeFootnotesDiv(self, root: etree.Element) -> etree.Element | None:
        """ Return `div` of footnotes as `etree` Element. """
        if not list(self.footnotes.keys()):
            return None
        div = etree.Element('div')
        div.set('class', 'footnote')
        etree.SubElement(div, 'hr')
        ol = etree.SubElement(div, 'ol')
        surrogate_parent = etree.Element('div')
        backlink_title = self.getConfig('BACKLINK_TITLE').replace('%d', '{}')
        for index, id in enumerate(self.footnotes.keys(), start=1):
            li = etree.SubElement(ol, 'li')
            li.set('id', self.makeFootnoteId(id))
            self.parser.parseChunk(surrogate_parent, self.footnotes[id])
            for el in list(surrogate_parent):
                li.append(el)
                surrogate_parent.remove(el)
            backlink = etree.Element('a')
            backlink.set('href', '#' + self.makeFootnoteRefId(id))
            backlink.set('class', 'footnote-backref')
            backlink.set('title', backlink_title.format(index))
            backlink.text = FN_BACKLINK_TEXT
            if len(li):
                node = li[-1]
                if node.tag == 'p':
                    node.text = node.text + NBSP_PLACEHOLDER
                    node.append(backlink)
                else:
                    p = etree.SubElement(li, 'p')
                    p.append(backlink)
        return div