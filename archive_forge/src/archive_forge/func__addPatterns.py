from __future__ import annotations
from . import Extension
from ..inlinepatterns import HtmlInlineProcessor, HTML_RE
from ..treeprocessors import InlineProcessor
from ..util import Registry
from typing import TYPE_CHECKING, Sequence
def _addPatterns(self, md: Markdown, patterns: Sequence[tuple[str, Sequence[int | str | etree.Element]]], serie: str, priority: int):
    for ind, pattern in enumerate(patterns):
        pattern += (md,)
        pattern = SubstituteTextPattern(*pattern)
        name = 'smarty-%s-%d' % (serie, ind)
        self.inlinePatterns.register(pattern, name, priority - ind)