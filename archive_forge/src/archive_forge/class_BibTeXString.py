from __future__ import absolute_import, unicode_literals
import re
import six
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.py3compat import fix_unicode_literals_in_doctest
from pybtex.utils import pairwise
from pybtex import py3compat
@py3compat.python_2_unicode_compatible
class BibTeXString(object):

    def __init__(self, chars, level=0, max_level=100):
        if level > max_level:
            raise BibTeXError('too many nested braces')
        self.level = level
        self.is_closed = False
        self.contents = list(self.find_closing_brace(iter(chars)))

    def __iter__(self):
        return self.traverse()

    def find_closing_brace(self, chars):
        for char in chars:
            if char == '{':
                yield BibTeXString(chars, self.level + 1)
            elif char == '}' and self.level > 0:
                self.is_closed = True
                return
            else:
                yield char

    def is_special_char(self):
        return self.level == 1 and self.contents and (self.contents[0] == '\\')

    def traverse(self, open=None, f=lambda char, string: char, close=None):
        if open is not None and self.level > 0:
            yield open(self)
        for child in self.contents:
            if hasattr(child, 'traverse'):
                if child.is_special_char():
                    if open is not None:
                        yield open(child)
                    yield f(child.inner_string(), child)
                    if close is not None:
                        yield close(child)
                else:
                    for result in child.traverse(open, f, close):
                        yield result
            else:
                yield f(child, self)
        if close is not None and self.level > 0 and self.is_closed:
            yield close(self)

    def __str__(self):
        return ''.join(self.traverse(open=lambda string: '{', close=lambda string: '}'))

    def inner_string(self):
        return ''.join((six.text_type(child) for child in self.contents))