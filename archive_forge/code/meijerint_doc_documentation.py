from __future__ import annotations
from typing import Any
from sympy.integrals.meijerint import _create_lookup_table
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.printing.latex import latex
 This module cooks up a docstring when imported. Its only purpose is to
    be displayed in the sphinx documentation. 