from typing import Dict, Type
from parso import tree
from parso.pgen2.generator import ReservedString
def _token_to_transition(grammar, type_, value):
    if type_.value.contains_syntax:
        try:
            return grammar.reserved_syntax_strings[value]
        except KeyError:
            pass
    return type_