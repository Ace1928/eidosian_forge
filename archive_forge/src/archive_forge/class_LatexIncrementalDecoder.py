import codecs
import dataclasses
import unicodedata
from typing import Optional, List, Union, Any, Iterator, Tuple, Type, Dict
from latexcodec import lexer
from codecs import CodecInfo
class LatexIncrementalDecoder(lexer.LatexIncrementalDecoder):
    """Translating incremental decoder for LaTeX."""
    table = _LATEX_UNICODE_TABLE
    token_buffer: List[lexer.Token]

    def __init__(self, errors='strict'):
        lexer.LatexIncrementalDecoder.__init__(self, errors=errors)

    def reset(self):
        lexer.LatexIncrementalDecoder.reset(self)
        self.token_buffer = []

    def getstate(self) -> Any:
        raise NotImplementedError

    def setstate(self, state: Any) -> None:
        raise NotImplementedError

    def get_unicode_tokens(self, chars: str, final: bool=False) -> Iterator[str]:
        for token in self.get_tokens(chars, final=final):
            self.token_buffer.append(token)
            for i in range(len(self.token_buffer), 0, -1):
                last_tokens = tuple(self.token_buffer[-i:])
                try:
                    unicode_text = self.table.unicode_map[last_tokens]
                except KeyError:
                    continue
                else:
                    for token2 in self.token_buffer[:-i]:
                        yield self.decode_token(token2)
                    yield unicode_text
                    self.token_buffer = []
                    break
            while len(self.token_buffer) >= self.table.max_length:
                yield self.decode_token(self.token_buffer.pop(0))
        if final:
            for token in self.token_buffer:
                yield self.decode_token(token)
            self.token_buffer = []