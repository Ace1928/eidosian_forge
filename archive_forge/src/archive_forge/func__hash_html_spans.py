import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _hash_html_spans(self, text):

    def _is_auto_link(s):
        if ':' in s and self._auto_link_re.match(s):
            return True
        elif '@' in s and self._auto_email_link_re.match(s):
            return True
        return False

    def _is_code_span(index, token):
        try:
            if token == '<code>':
                peek_tokens = split_tokens[index:index + 3]
            elif token == '</code>':
                peek_tokens = split_tokens[index - 2:index + 1]
            else:
                return False
        except IndexError:
            return False
        return re.match('<code>md5-[A-Fa-f0-9]{32}</code>', ''.join(peek_tokens))
    tokens = []
    split_tokens = self._sorta_html_tokenize_re.split(text)
    is_html_markup = False
    for index, token in enumerate(split_tokens):
        if is_html_markup and (not _is_auto_link(token)) and (not _is_code_span(index, token)):
            sanitized = self._sanitize_html(token)
            key = _hash_text(sanitized)
            self.html_spans[key] = sanitized
            tokens.append(key)
        else:
            tokens.append(self._encode_incomplete_tags(token))
        is_html_markup = not is_html_markup
    return ''.join(tokens)