from sympy.core.function import Derivative
from sympy.core.function import UndefinedFunction, AppliedUndef
from sympy.core.symbol import Symbol
from sympy.interactive.printing import init_printing
from sympy.printing.latex import LatexPrinter
from sympy.printing.pretty.pretty import PrettyPrinter
from sympy.printing.pretty.pretty_symbology import center_accent
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import PRECEDENCE
class VectorLatexPrinter(LatexPrinter):
    """Latex Printer for vector expressions. """

    def _print_Function(self, expr, exp=None):
        from sympy.physics.vector.functions import dynamicsymbols
        func = expr.func.__name__
        t = dynamicsymbols._t
        if hasattr(self, '_print_' + func) and (not isinstance(type(expr), UndefinedFunction)):
            return getattr(self, '_print_' + func)(expr, exp)
        elif isinstance(type(expr), UndefinedFunction) and expr.args == (t,):
            expr = Symbol(func)
            if exp is not None:
                base = self.parenthesize(expr, PRECEDENCE['Pow'])
                base = self.parenthesize_super(base)
                return '%s^{%s}' % (base, exp)
            else:
                return super()._print(expr)
        else:
            return super()._print_Function(expr, exp)

    def _print_Derivative(self, der_expr):
        from sympy.physics.vector.functions import dynamicsymbols
        der_expr = der_expr.doit()
        if not isinstance(der_expr, Derivative):
            return '\\left(%s\\right)' % self.doprint(der_expr)
        t = dynamicsymbols._t
        expr = der_expr.expr
        red = expr.atoms(AppliedUndef)
        syms = der_expr.variables
        test1 = not all((True for i in red if i.free_symbols == {t}))
        test2 = not all((t == i for i in syms))
        if test1 or test2:
            return super()._print_Derivative(der_expr)
        dots = len(syms)
        base = self._print_Function(expr)
        base_split = base.split('_', 1)
        base = base_split[0]
        if dots == 1:
            base = '\\dot{%s}' % base
        elif dots == 2:
            base = '\\ddot{%s}' % base
        elif dots == 3:
            base = '\\dddot{%s}' % base
        elif dots == 4:
            base = '\\ddddot{%s}' % base
        else:
            return super()._print_Derivative(der_expr)
        if len(base_split) != 1:
            base += '_' + base_split[1]
        return base