from .utils import logger, NO_VALUE
from typing import Mapping, Iterable, Callable, Union, TypeVar, Tuple, Any, List, Set, Optional, Collection, TYPE_CHECKING
class UnexpectedEOF(ParseError, UnexpectedInput):
    """An exception that is raised by the parser, when the input ends while it still expects a token.
    """
    expected: 'List[Token]'

    def __init__(self, expected, state=None, terminals_by_name=None):
        super(UnexpectedEOF, self).__init__()
        self.expected = expected
        self.state = state
        from .lexer import Token
        self.token = Token('<EOF>', '')
        self.pos_in_stream = -1
        self.line = -1
        self.column = -1
        self._terminals_by_name = terminals_by_name

    def __str__(self):
        message = 'Unexpected end-of-input. '
        message += self._format_expected(self.expected)
        return message