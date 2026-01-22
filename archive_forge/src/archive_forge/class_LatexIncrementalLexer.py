import codecs
import re
import unicodedata
from abc import ABC, ABCMeta
from typing import Iterator, Tuple, Sequence, Any, NamedTuple
class LatexIncrementalLexer(LatexLexer, ABC):
    """A very simple incremental lexer for tex/latex code. Roughly
    follows the state machine described in Tex By Topic, Chapter 2.

    The generated tokens satisfy:

    * no newline characters: paragraphs are separated by '\\par'
    * spaces following control tokens are compressed
    """
    partoken = Token(u'control_word', u'\\par')
    spacetoken = Token(u'space', u' ')
    replacetoken = Token(u'chars', u'�')
    curlylefttoken = Token(u'chars', u'{')
    curlyrighttoken = Token(u'chars', u'}')
    state: str
    inline_math: bool

    def reset(self) -> None:
        super().reset()
        self.state = 'N'
        self.inline_math = False

    def getstate(self) -> Any:
        return (self.raw_buffer, {'M': 0, 'N': 1, 'S': 2}[self.state] | (4 if self.inline_math else 0))

    def setstate(self, state: Any):
        self.raw_buffer = state[0]
        self.state = {0: 'M', 1: 'N', 2: 'S'}[state[1] & 3]
        self.inline_math = bool(state[1] & 4)

    def get_tokens(self, chars: str, final: bool=False) -> Iterator[Token]:
        """Yield tokens while maintaining a state. Also skip
        whitespace after control words and (some) control symbols.
        Replaces newlines by spaces and \\par commands depending on
        the context.
        """
        pos = -len(self.raw_buffer.text)
        for token in self.get_raw_tokens(chars, final=final):
            pos = pos + len(token.text)
            assert pos >= 0
            if token.name == 'newline':
                if self.state == 'N':
                    yield self.partoken
                elif self.state == 'S':
                    self.state = 'N'
                elif self.state == 'M':
                    self.state = 'N'
                    yield self.spacetoken
                else:
                    raise AssertionError('unknown tex state {0!r}'.format(self.state))
            elif token.name == 'space':
                if self.state == 'N':
                    pass
                elif self.state == 'S':
                    pass
                elif self.state == 'M':
                    self.state = 'S'
                    yield token
                else:
                    raise AssertionError('unknown state {0!r}'.format(self.state))
            elif token.name == 'mathshift':
                self.inline_math = not self.inline_math
                self.state = 'M'
                yield token
            elif token.name == 'parameter':
                self.state = 'M'
                yield token
            elif token.name == 'control_word':
                self.state = 'S'
                yield token
            elif token.name == 'control_symbol':
                self.state = 'S'
                yield token
            elif token.name == 'control_symbol_x' or token.name == 'control_symbol_x2':
                self.state = 'M'
                yield token
            elif token.name == 'comment':
                self.state = 'S'
            elif token.name == 'chars':
                self.state = 'M'
                yield token
            elif token.name == 'unknown':
                if self.errors == 'strict':
                    raise UnicodeDecodeError('latex', chars.encode('utf8'), pos - len(token.text), pos, 'unknown token {0!r}'.format(token.text))
                elif self.errors == 'ignore':
                    pass
                elif self.errors == 'replace':
                    yield self.replacetoken
                else:
                    raise NotImplementedError('error mode {0!r} not supported'.format(self.errors))
            else:
                raise AssertionError('unknown token name {0!r}'.format(token.name))