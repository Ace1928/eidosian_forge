from __future__ import annotations
from . import Extension
from ..treeprocessors import Treeprocessor
from ..util import parseBoolValue
from typing import TYPE_CHECKING, Callable, Any
class HiliteTreeprocessor(Treeprocessor):
    """ Highlight source code in code blocks. """
    config: dict[str, Any]

    def code_unescape(self, text: str) -> str:
        """Unescape code."""
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&amp;', '&')
        return text

    def run(self, root: etree.Element) -> None:
        """ Find code blocks and store in `htmlStash`. """
        blocks = root.iter('pre')
        for block in blocks:
            if len(block) == 1 and block[0].tag == 'code':
                local_config = self.config.copy()
                text = block[0].text
                if text is None:
                    continue
                code = CodeHilite(self.code_unescape(text), tab_length=self.md.tab_length, style=local_config.pop('pygments_style', 'default'), **local_config)
                placeholder = self.md.htmlStash.store(code.hilite())
                block.clear()
                block.tag = 'p'
                block.text = placeholder