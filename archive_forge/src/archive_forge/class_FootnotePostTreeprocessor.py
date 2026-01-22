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
class FootnotePostTreeprocessor(Treeprocessor):
    """ Amend footnote div with duplicates. """

    def __init__(self, footnotes: FootnoteExtension):
        self.footnotes = footnotes

    def add_duplicates(self, li: etree.Element, duplicates: int) -> None:
        """ Adjust current `li` and add the duplicates: `fnref2`, `fnref3`, etc. """
        for link in li.iter('a'):
            if link.attrib.get('class', '') == 'footnote-backref':
                ref, rest = link.attrib['href'].split(self.footnotes.get_separator(), 1)
                links = []
                for index in range(2, duplicates + 1):
                    sib_link = copy.deepcopy(link)
                    sib_link.attrib['href'] = '%s%d%s%s' % (ref, index, self.footnotes.get_separator(), rest)
                    links.append(sib_link)
                    self.offset += 1
                el = list(li)[-1]
                for link in links:
                    el.append(link)
                break

    def get_num_duplicates(self, li: etree.Element) -> int:
        """ Get the number of duplicate refs of the footnote. """
        fn, rest = li.attrib.get('id', '').split(self.footnotes.get_separator(), 1)
        link_id = '{}ref{}{}'.format(fn, self.footnotes.get_separator(), rest)
        return self.footnotes.found_refs.get(link_id, 0)

    def handle_duplicates(self, parent: etree.Element) -> None:
        """ Find duplicate footnotes and format and add the duplicates. """
        for li in list(parent):
            count = self.get_num_duplicates(li)
            if count > 1:
                self.add_duplicates(li, count)

    def run(self, root: etree.Element) -> None:
        """ Crawl the footnote div and add missing duplicate footnotes. """
        self.offset = 0
        for div in root.iter('div'):
            if div.attrib.get('class', '') == 'footnote':
                for ol in div.iter('ol'):
                    self.handle_duplicates(ol)
                    break