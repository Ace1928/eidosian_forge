import codecs
import re
import unicodedata
from abc import ABC, ABCMeta
from typing import Iterator, Tuple, Sequence, Any, NamedTuple
class RegexpLexer(codecs.IncrementalDecoder, metaclass=MetaRegexpLexer):
    """Abstract base class for regexp based lexers."""
    emptytoken = Token('unknown', '')
    tokens: Sequence[Tuple[str, str]] = ()
    errors: str
    raw_buffer: Token
    regexp: Any

    def __init__(self, errors: str='strict') -> None:
        """Initialize the codec."""
        super().__init__(errors=errors)
        self.errors = errors
        self.reset()

    def reset(self) -> None:
        """Reset state."""
        self.raw_buffer = self.emptytoken

    def getstate(self) -> Any:
        """Get state."""
        return (self.raw_buffer.text, 0)

    def setstate(self, state: Any) -> None:
        """Set state. The *state* must correspond to the return value
        of a previous :meth:`getstate` call.
        """
        self.raw_buffer = Token('unknown', state[0])

    def get_raw_tokens(self, chars: str, final: bool=False) -> Iterator[Token]:
        """Yield tokens without any further processing. Tokens are one of:

        - ``\\<word>``: a control word (i.e. a command)
        - ``\\<symbol>``: a control symbol (i.e. \\^ etc.)
        - ``#<n>``: a parameter
        - a series of byte characters
        """
        if self.raw_buffer.text:
            chars = self.raw_buffer.text + chars
        self.raw_buffer = self.emptytoken
        for match in self.regexp.finditer(chars):
            if self.raw_buffer.text:
                yield self.raw_buffer
            assert match.lastgroup is not None
            self.raw_buffer = Token(match.lastgroup, match.group(0))
        if final:
            for token in self.flush_raw_tokens():
                yield token

    def flush_raw_tokens(self) -> Iterator[Token]:
        """Flush the raw token buffer."""
        if self.raw_buffer.text:
            yield self.raw_buffer
            self.raw_buffer = self.emptytoken