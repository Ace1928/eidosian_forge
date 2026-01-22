import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _strict_tag_block_sub(self, text, html_tags_re, callback, allow_indent=False):
    """
        Finds and substitutes HTML blocks within blocks of text

        Args:
            text: the text to search
            html_tags_re: a regex pattern of HTML block tags to match against.
                For example, `Markdown._block_tags_a`
            callback: callback function that receives the found HTML text block
            allow_indent: allow matching HTML blocks that are not completely outdented
        """
    tag_count = 0
    current_tag = html_tags_re
    block = ''
    result = ''
    for chunk in text.splitlines(True):
        is_markup = re.match('^(\\s{0,%s})(?:</code>(?=</pre>))?(</?(%s)\\b>?)' % ('' if allow_indent else '0', current_tag), chunk)
        block += chunk
        if is_markup:
            if chunk.startswith('%s</' % is_markup.group(1)):
                tag_count -= 1
            elif self._tag_is_closed(is_markup.group(3), chunk):
                is_markup = None
            else:
                tag_count += 1
                current_tag = is_markup.group(3)
        if tag_count == 0:
            if is_markup:
                block = callback(block.rstrip('\n'))
            current_tag = html_tags_re
            result += block
            block = ''
    result += block
    return result