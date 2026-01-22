import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _hash_html_blocks(self, text, raw=False):
    """Hashify HTML blocks

        We only want to do this for block-level HTML tags, such as headers,
        lists, and tables. That's because we still want to wrap <p>s around
        "paragraphs" that are wrapped in non-block-level tags, such as anchors,
        phrase emphasis, and spans. The list of tags we're looking for is
        hard-coded.

        @param raw {boolean} indicates if these are raw HTML blocks in
            the original source. It makes a difference in "safe" mode.
        """
    if '<' not in text:
        return text
    hash_html_block_sub = _curry(self._hash_html_block_sub, raw=raw)
    text = self._strict_tag_block_sub(text, self._block_tags_a, hash_html_block_sub)
    text = self._liberal_tag_block_re.sub(hash_html_block_sub, text)
    if '<hr' in text:
        _hr_tag_re = _hr_tag_re_from_tab_width(self.tab_width)
        text = _hr_tag_re.sub(hash_html_block_sub, text)
    if '<!--' in text:
        start = 0
        while True:
            try:
                start_idx = text.index('<!--', start)
            except ValueError:
                break
            try:
                end_idx = text.index('-->', start_idx) + 3
            except ValueError:
                break
            start = end_idx
            if start_idx:
                for i in range(self.tab_width - 1):
                    if text[start_idx - 1] != ' ':
                        break
                    start_idx -= 1
                    if start_idx == 0:
                        break
                if start_idx == 0:
                    pass
                elif start_idx == 1 and text[0] == '\n':
                    start_idx = 0
                elif text[start_idx - 2:start_idx] == '\n\n':
                    pass
                else:
                    break
            while end_idx < len(text):
                if text[end_idx] not in ' \t':
                    break
                end_idx += 1
            if text[end_idx:end_idx + 2] not in ('', '\n', '\n\n'):
                continue
            html = text[start_idx:end_idx]
            if raw and self.safe_mode:
                html = self._sanitize_html(html)
            key = _hash_text(html)
            self.html_blocks[key] = html
            text = text[:start_idx] + '\n\n' + key + '\n\n' + text[end_idx:]
    if 'xml' in self.extras:
        _xml_oneliner_re = _xml_oneliner_re_from_tab_width(self.tab_width)
        text = _xml_oneliner_re.sub(hash_html_block_sub, text)
    return text