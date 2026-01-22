import re
import sys
import time
from pygments.filter import apply_filters, Filter
from pygments.filters import get_filter_by_name
from pygments.token import Error, Text, Other, Whitespace, _TokenType
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
from pygments.regexopt import regex_opt
def _preprocess_lexer_input(self, text):
    """Apply preprocessing such as decoding the input, removing BOM and normalizing newlines."""
    if not isinstance(text, str):
        if self.encoding == 'guess':
            text, _ = guess_decode(text)
        elif self.encoding == 'chardet':
            try:
                import chardet
            except ImportError as e:
                raise ImportError('To enable chardet encoding guessing, please install the chardet library from http://chardet.feedparser.org/') from e
            decoded = None
            for bom, encoding in _encoding_map:
                if text.startswith(bom):
                    decoded = text[len(bom):].decode(encoding, 'replace')
                    break
            if decoded is None:
                enc = chardet.detect(text[:1024])
                decoded = text.decode(enc.get('encoding') or 'utf-8', 'replace')
            text = decoded
        else:
            text = text.decode(self.encoding)
            if text.startswith('\ufeff'):
                text = text[len('\ufeff'):]
    elif text.startswith('\ufeff'):
        text = text[len('\ufeff'):]
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')
    if self.stripall:
        text = text.strip()
    elif self.stripnl:
        text = text.strip('\n')
    if self.tabsize > 0:
        text = text.expandtabs(self.tabsize)
    if self.ensurenl and (not text.endswith('\n')):
        text += '\n'
    return text