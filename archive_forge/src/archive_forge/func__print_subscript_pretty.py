from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.core.containers import Tuple
from sympy.utilities.iterables import is_sequence
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.matrixutils import (
def _print_subscript_pretty(self, a, b):
    top = prettyForm(*b.left(' ' * a.width()))
    bot = prettyForm(*a.right(' ' * b.width()))
    return prettyForm(*bot.below(top), binding=prettyForm.POW)