from __future__ import annotations
from collections import OrderedDict
from typing import TYPE_CHECKING, Any
from . import util
import re
class RawHtmlPostprocessor(Postprocessor):
    """ Restore raw html to the document. """
    BLOCK_LEVEL_REGEX = re.compile('^\\<\\/?([^ >]+)')

    def run(self, text: str) -> str:
        """ Iterate over html stash and restore html. """
        replacements = OrderedDict()
        for i in range(self.md.htmlStash.html_counter):
            html = self.stash_to_string(self.md.htmlStash.rawHtmlBlocks[i])
            if self.isblocklevel(html):
                replacements['<p>{}</p>'.format(self.md.htmlStash.get_placeholder(i))] = html
            replacements[self.md.htmlStash.get_placeholder(i)] = html

        def substitute_match(m: re.Match[str]) -> str:
            key = m.group(0)
            if key not in replacements:
                if key[3:-4] in replacements:
                    return f'<p>{replacements[key[3:-4]]}</p>'
                else:
                    return key
            return replacements[key]
        if replacements:
            base_placeholder = util.HTML_PLACEHOLDER % '([0-9]+)'
            pattern = re.compile(f'<p>{base_placeholder}</p>|{base_placeholder}')
            processed_text = pattern.sub(substitute_match, text)
        else:
            return text
        if processed_text == text:
            return processed_text
        else:
            return self.run(processed_text)

    def isblocklevel(self, html: str) -> bool:
        """ Check is block of HTML is block-level. """
        m = self.BLOCK_LEVEL_REGEX.match(html)
        if m:
            if m.group(1)[0] in ('!', '?', '@', '%'):
                return True
            return self.md.is_block_level(m.group(1))
        return False

    def stash_to_string(self, text: str) -> str:
        """ Convert a stashed object to a string. """
        return str(text)