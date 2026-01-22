from __future__ import division
from pygments.formatter import Formatter
from pygments.lexer import Lexer
from pygments.token import Token, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, StringIO, xrange, \
class LatexEmbeddedLexer(Lexer):
    """
    This lexer takes one lexer as argument, the lexer for the language
    being formatted, and the left and right delimiters for escaped text.

    First everything is scanned using the language lexer to obtain
    strings and comments. All other consecutive tokens are merged and
    the resulting text is scanned for escaped segments, which are given
    the Token.Escape type. Finally text that is not escaped is scanned
    again with the language lexer.
    """

    def __init__(self, left, right, lang, **options):
        self.left = left
        self.right = right
        self.lang = lang
        Lexer.__init__(self, **options)

    def get_tokens_unprocessed(self, text):
        buf = ''
        idx = 0
        for i, t, v in self.lang.get_tokens_unprocessed(text):
            if t in Token.Comment or t in Token.String:
                if buf:
                    for x in self.get_tokens_aux(idx, buf):
                        yield x
                    buf = ''
                yield (i, t, v)
            else:
                if not buf:
                    idx = i
                buf += v
        if buf:
            for x in self.get_tokens_aux(idx, buf):
                yield x

    def get_tokens_aux(self, index, text):
        while text:
            a, sep1, text = text.partition(self.left)
            if a:
                for i, t, v in self.lang.get_tokens_unprocessed(a):
                    yield (index + i, t, v)
                    index += len(a)
            if sep1:
                b, sep2, text = text.partition(self.right)
                if sep2:
                    yield (index + len(sep1), Token.Escape, b)
                    index += len(sep1) + len(b) + len(sep2)
                else:
                    yield (index, Token.Error, sep1)
                    index += len(sep1)
                    text = b