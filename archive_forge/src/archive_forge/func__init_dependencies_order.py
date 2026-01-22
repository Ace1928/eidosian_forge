from __future__ import annotations
from sympy.core import Basic, sympify
from sympy.polys.polyerrors import GeneratorsError, OptionError, FlagError
from sympy.utilities import numbered_symbols, topological_sort, public
from sympy.utilities.iterables import has_dups, is_sequence
import sympy.polys
import re
@classmethod
def _init_dependencies_order(cls):
    """Resolve the order of options' processing. """
    if cls.__order__ is None:
        vertices, edges = ([], set())
        for name, option in cls.__options__.items():
            vertices.append(name)
            for _name in option.after:
                edges.add((_name, name))
            for _name in option.before:
                edges.add((name, _name))
        try:
            cls.__order__ = topological_sort((vertices, list(edges)))
        except ValueError:
            raise RuntimeError('cycle detected in sympy.polys options framework')