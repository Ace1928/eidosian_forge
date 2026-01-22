from ast import literal_eval
from typing import TypeVar, Generic, Mapping, Sequence, Set, Union
from parso.pgen2.grammar_parser import GrammarParser, NFAState
def _make_transition(token_namespace, reserved_syntax_strings, label):
    """
    Creates a reserved string ("if", "for", "*", ...) or returns the token type
    (NUMBER, STRING, ...) for a given grammar terminal.
    """
    if label[0].isalpha():
        return getattr(token_namespace, label)
    else:
        assert label[0] in ('"', "'"), label
        assert not label.startswith('"""') and (not label.startswith("'''"))
        value = literal_eval(label)
        try:
            return reserved_syntax_strings[value]
        except KeyError:
            r = reserved_syntax_strings[value] = ReservedString(value)
            return r