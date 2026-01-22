from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
class BacktickInlineProcessor(InlineProcessor):
    """ Return a `<code>` element containing the escaped matching text. """

    def __init__(self, pattern: str):
        InlineProcessor.__init__(self, pattern)
        self.ESCAPED_BSLASH = '{}{}{}'.format(util.STX, ord('\\'), util.ETX)
        self.tag = 'code'
        ' The tag of the rendered element. '

    def handleMatch(self, m: re.Match[str], data: str) -> tuple[etree.Element | str, int, int]:
        """
        If the match contains `group(3)` of a pattern, then return a `code`
        [`Element`][xml.etree.ElementTree.Element] which contains HTML escaped text (with
        [`code_escape`][markdown.util.code_escape]) as an [`AtomicString`][markdown.util.AtomicString].

        If the match does not contain `group(3)` then return the text of `group(1)` backslash escaped.

        """
        if m.group(3):
            el = etree.Element(self.tag)
            el.text = util.AtomicString(util.code_escape(m.group(3).strip()))
            return (el, m.start(0), m.end(0))
        else:
            return (m.group(1).replace('\\\\', self.ESCAPED_BSLASH), m.start(0), m.end(0))