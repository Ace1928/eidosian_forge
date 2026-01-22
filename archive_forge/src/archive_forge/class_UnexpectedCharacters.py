from .utils import logger, NO_VALUE
from typing import Mapping, Iterable, Callable, Union, TypeVar, Tuple, Any, List, Set, Optional, Collection, TYPE_CHECKING
class UnexpectedCharacters(LexError, UnexpectedInput):
    """An exception that is raised by the lexer, when it cannot match the next
    string of characters to any of its terminals.
    """
    allowed: Set[str]
    considered_tokens: Set[Any]

    def __init__(self, seq, lex_pos, line, column, allowed=None, considered_tokens=None, state=None, token_history=None, terminals_by_name=None, considered_rules=None):
        super(UnexpectedCharacters, self).__init__()
        self.line = line
        self.column = column
        self.pos_in_stream = lex_pos
        self.state = state
        self._terminals_by_name = terminals_by_name
        self.allowed = allowed
        self.considered_tokens = considered_tokens
        self.considered_rules = considered_rules
        self.token_history = token_history
        if isinstance(seq, bytes):
            self.char = seq[lex_pos:lex_pos + 1].decode('ascii', 'backslashreplace')
        else:
            self.char = seq[lex_pos]
        self._context = self.get_context(seq)

    def __str__(self):
        message = "No terminal matches '%s' in the current parser context, at line %d col %d" % (self.char, self.line, self.column)
        message += '\n\n' + self._context
        if self.allowed:
            message += self._format_expected(self.allowed)
        if self.token_history:
            message += '\nPrevious tokens: %s\n' % ', '.join((repr(t) for t in self.token_history))
        return message