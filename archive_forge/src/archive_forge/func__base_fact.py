from collections import defaultdict
from typing import Iterator
from .logic import Logic, And, Or, Not
def _base_fact(atom):
    """Return the literal fact of an atom.

    Effectively, this merely strips the Not around a fact.
    """
    if isinstance(atom, Not):
        return atom.arg
    else:
        return atom