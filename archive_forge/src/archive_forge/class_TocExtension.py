from __future__ import annotations
from . import Extension
from ..treeprocessors import Treeprocessor
from ..util import parseBoolValue, AMP_SUBSTITUTE, deprecated, HTML_PLACEHOLDER_RE, AtomicString
from ..treeprocessors import UnescapeTreeprocessor
from ..serializers import RE_AMP
import re
import html
import unicodedata
from copy import deepcopy
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any, Iterator, MutableSet
class TocExtension(Extension):
    TreeProcessorClass = TocTreeprocessor

    def __init__(self, **kwargs):
        self.config = {'marker': ['[TOC]', 'Text to find and replace with Table of Contents. Set to an empty string to disable. Default: `[TOC]`.'], 'title': ['', 'Title to insert into TOC `<div>`. Default: an empty string.'], 'title_class': ['toctitle', 'CSS class used for the title. Default: `toctitle`.'], 'toc_class': ['toc', 'CSS class(es) used for the link. Default: `toclink`.'], 'anchorlink': [False, 'True if header should be a self link. Default: `False`.'], 'anchorlink_class': ['toclink', 'CSS class(es) used for the link. Defaults: `toclink`.'], 'permalink': [0, 'True or link text if a Sphinx-style permalink should be added. Default: `False`.'], 'permalink_class': ['headerlink', 'CSS class(es) used for the link. Default: `headerlink`.'], 'permalink_title': ['Permanent link', 'Title attribute of the permalink. Default: `Permanent link`.'], 'permalink_leading': [False, 'True if permalinks should be placed at start of the header, rather than end. Default: False.'], 'baselevel': ['1', 'Base level for headers. Default: `1`.'], 'slugify': [slugify, 'Function to generate anchors based on header text. Default: `slugify`.'], 'separator': ['-', 'Word separator. Default: `-`.'], 'toc_depth': [6, 'Define the range of section levels to include in the Table of Contents. A single integer (b) defines the bottom section level (<h1>..<hb>) only. A string consisting of two digits separated by a hyphen in between (`2-5`) defines the top (t) and the bottom (b) (<ht>..<hb>). Default: `6` (bottom).']}
        ' Default configuration options. '
        super().__init__(**kwargs)

    def extendMarkdown(self, md):
        """ Add TOC tree processor to Markdown. """
        md.registerExtension(self)
        self.md = md
        self.reset()
        tocext = self.TreeProcessorClass(md, self.getConfigs())
        md.treeprocessors.register(tocext, 'toc', 5)

    def reset(self) -> None:
        self.md.toc = ''
        self.md.toc_tokens = []