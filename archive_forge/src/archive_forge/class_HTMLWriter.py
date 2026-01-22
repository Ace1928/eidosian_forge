import keyword
import tokenize
from html import escape
from typing import List
from . import reflect
class HTMLWriter:
    """
    Write the stream of tokens and whitespace from L{TokenPrinter}, formating
    tokens as HTML spans.
    """
    noSpan: List[str] = []

    def __init__(self, writer):
        self.writer = writer
        noSpan: List[str] = []
        reflect.accumulateClassList(self.__class__, 'noSpan', noSpan)
        self.noSpan = noSpan

    def write(self, token, type=None):
        if isinstance(token, bytes):
            token = token.decode('utf-8')
        token = escape(token)
        token = token.encode('utf-8')
        if type is None or type in self.noSpan:
            self.writer(token)
        else:
            self.writer(b'<span class="py-src-' + type.encode('utf-8') + b'">' + token + b'</span>')