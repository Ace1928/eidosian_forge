from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
def _arrayify(self, indexed):
    from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
    try:
        return convert_indexed_to_array(indexed)
    except Exception:
        return indexed