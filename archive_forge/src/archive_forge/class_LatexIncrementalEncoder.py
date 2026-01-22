import codecs
import dataclasses
import unicodedata
from typing import Optional, List, Union, Any, Iterator, Tuple, Type, Dict
from latexcodec import lexer
from codecs import CodecInfo
class LatexIncrementalEncoder(lexer.LatexIncrementalEncoder):
    """Translating incremental encoder for latex. Maintains a state to
    determine whether control spaces etc. need to be inserted.
    """
    emptytoken = lexer.Token('unknown', '')
    table = _LATEX_UNICODE_TABLE
    state: str

    def __init__(self, errors='strict'):
        super().__init__(errors=errors)
        self.reset()

    def reset(self):
        super(LatexIncrementalEncoder, self).reset()
        self.state = 'M'

    def get_space_bytes(self, bytes_: str) -> Tuple[str, str]:
        """Inserts space bytes in space eating mode."""
        if self.state == 'S':
            if bytes_.startswith(' '):
                return ('\\ ', bytes_[1:])
            else:
                return (' ', bytes_)
        else:
            return ('', bytes_)

    def _get_latex_chars_tokens_from_char(self, c: str) -> Tuple[str, Tuple[lexer.Token, ...]]:
        if ord(c) < 128:
            try:
                return self.table.latex_map[c]
            except KeyError:
                pass
        try:
            c.encode(self.inputenc, 'strict')
        except UnicodeEncodeError:
            pass
        else:
            return (c, (lexer.Token(name='chars', text=c),))
        try:
            return self.table.latex_map[c]
        except KeyError:
            if self.errors == 'strict':
                raise UnicodeEncodeError('latex', c, 0, 1, "don't know how to translate {0} into latex".format(repr(c)))
            elif self.errors == 'ignore':
                return ('', (self.emptytoken,))
            elif self.errors == 'replace':
                bytes_ = '{\\char' + str(ord(c)) + '}'
                return (bytes_, (lexer.Token(name='chars', text=bytes_),))
            elif self.errors == 'keep':
                return (c, (lexer.Token(name='chars', text=c),))
            else:
                raise ValueError('latex codec does not support {0} errors'.format(self.errors))

    def get_latex_chars(self, unicode_: str, final: bool=False) -> Iterator[str]:
        if not isinstance(unicode_, str):
            raise TypeError('expected unicode for encode input, but got {0} instead'.format(unicode_.__class__.__name__))
        for pos, c in enumerate(unicode_):
            bytes_, tokens = self._get_latex_chars_tokens_from_char(c)
            space, bytes_ = self.get_space_bytes(bytes_)
            if tokens and tokens[-1].name == 'control_word':
                self.state = 'S'
            elif tokens:
                self.state = 'M'
            if space:
                yield space
            yield bytes_