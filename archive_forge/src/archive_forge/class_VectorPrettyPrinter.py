from sympy.core.function import Derivative
from sympy.core.function import UndefinedFunction, AppliedUndef
from sympy.core.symbol import Symbol
from sympy.interactive.printing import init_printing
from sympy.printing.latex import LatexPrinter
from sympy.printing.pretty.pretty import PrettyPrinter
from sympy.printing.pretty.pretty_symbology import center_accent
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import PRECEDENCE
class VectorPrettyPrinter(PrettyPrinter):
    """Pretty Printer for vectorialexpressions. """

    def _print_Derivative(self, deriv):
        from sympy.physics.vector.functions import dynamicsymbols
        t = dynamicsymbols._t
        dot_i = 0
        syms = list(reversed(deriv.variables))
        while len(syms) > 0:
            if syms[-1] == t:
                syms.pop()
                dot_i += 1
            else:
                return super()._print_Derivative(deriv)
        if not (isinstance(type(deriv.expr), UndefinedFunction) and deriv.expr.args == (t,)):
            return super()._print_Derivative(deriv)
        else:
            pform = self._print_Function(deriv.expr)
        if len(pform.picture) > 1:
            return super()._print_Derivative(deriv)
        if dot_i >= 5:
            return super()._print_Derivative(deriv)
        dots = {0: '', 1: '̇', 2: '̈', 3: '⃛', 4: '⃜'}
        d = pform.__dict__
        if not self._use_unicode:
            apostrophes = ''
            for i in range(0, dot_i):
                apostrophes += "'"
            d['picture'][0] += apostrophes + '(t)'
        else:
            d['picture'] = [center_accent(d['picture'][0], dots[dot_i])]
        return pform

    def _print_Function(self, e):
        from sympy.physics.vector.functions import dynamicsymbols
        t = dynamicsymbols._t
        func = e.func
        args = e.args
        func_name = func.__name__
        pform = self._print_Symbol(Symbol(func_name))
        if not (isinstance(func, UndefinedFunction) and args == (t,)):
            return super()._print_Function(e)
        return pform