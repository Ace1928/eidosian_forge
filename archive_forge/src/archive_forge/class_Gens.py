from __future__ import annotations
from sympy.core import Basic, sympify
from sympy.polys.polyerrors import GeneratorsError, OptionError, FlagError
from sympy.utilities import numbered_symbols, topological_sort, public
from sympy.utilities.iterables import has_dups, is_sequence
import sympy.polys
import re
class Gens(Option, metaclass=OptionType):
    """``gens`` option to polynomial manipulation functions. """
    option = 'gens'
    requires: list[str] = []
    excludes: list[str] = []

    @classmethod
    def default(cls):
        return ()

    @classmethod
    def preprocess(cls, gens):
        if isinstance(gens, Basic):
            gens = (gens,)
        elif len(gens) == 1 and is_sequence(gens[0]):
            gens = gens[0]
        if gens == (None,):
            gens = ()
        elif has_dups(gens):
            raise GeneratorsError('duplicated generators: %s' % str(gens))
        elif any((gen.is_commutative is False for gen in gens)):
            raise GeneratorsError('non-commutative generators: %s' % str(gens))
        return tuple(gens)