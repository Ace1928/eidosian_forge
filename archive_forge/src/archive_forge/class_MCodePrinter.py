from __future__ import annotations
from typing import Any
from sympy.core import Basic, Expr, Float
from sympy.core.sorting import default_sort_key
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence
class MCodePrinter(CodePrinter):
    """A printer to convert Python expressions to
    strings of the Wolfram's Mathematica code
    """
    printmethod = '_mcode'
    language = 'Wolfram Language'
    _default_settings: dict[str, Any] = {'order': None, 'full_prec': 'auto', 'precision': 15, 'user_functions': {}, 'human': True, 'allow_unknown_functions': False}
    _number_symbols: set[tuple[Expr, Float]] = set()
    _not_supported: set[Basic] = set()

    def __init__(self, settings={}):
        """Register function mappings supplied by user"""
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {}).copy()
        for k, v in userfuncs.items():
            if not isinstance(v, list):
                userfuncs[k] = [(lambda *x: True, v)]
        self.known_functions.update(userfuncs)

    def _format_code(self, lines):
        return lines

    def _print_Pow(self, expr):
        PREC = precedence(expr)
        return '%s^%s' % (self.parenthesize(expr.base, PREC), self.parenthesize(expr.exp, PREC))

    def _print_Mul(self, expr):
        PREC = precedence(expr)
        c, nc = expr.args_cnc()
        res = super()._print_Mul(expr.func(*c))
        if nc:
            res += '*'
            res += '**'.join((self.parenthesize(a, PREC) for a in nc))
        return res

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return '{} {} {}'.format(lhs_code, op, rhs_code)

    def _print_Zero(self, expr):
        return '0'

    def _print_One(self, expr):
        return '1'

    def _print_NegativeOne(self, expr):
        return '-1'

    def _print_Half(self, expr):
        return '1/2'

    def _print_ImaginaryUnit(self, expr):
        return 'I'

    def _print_Infinity(self, expr):
        return 'Infinity'

    def _print_NegativeInfinity(self, expr):
        return '-Infinity'

    def _print_ComplexInfinity(self, expr):
        return 'ComplexInfinity'

    def _print_NaN(self, expr):
        return 'Indeterminate'

    def _print_Exp1(self, expr):
        return 'E'

    def _print_Pi(self, expr):
        return 'Pi'

    def _print_GoldenRatio(self, expr):
        return 'GoldenRatio'

    def _print_TribonacciConstant(self, expr):
        expanded = expr.expand(func=True)
        PREC = precedence(expr)
        return self.parenthesize(expanded, PREC)

    def _print_EulerGamma(self, expr):
        return 'EulerGamma'

    def _print_Catalan(self, expr):
        return 'Catalan'

    def _print_list(self, expr):
        return '{' + ', '.join((self.doprint(a) for a in expr)) + '}'
    _print_tuple = _print_list
    _print_Tuple = _print_list

    def _print_ImmutableDenseMatrix(self, expr):
        return self.doprint(expr.tolist())

    def _print_ImmutableSparseMatrix(self, expr):

        def print_rule(pos, val):
            return '{} -> {}'.format(self.doprint((pos[0] + 1, pos[1] + 1)), self.doprint(val))

        def print_data():
            items = sorted(expr.todok().items(), key=default_sort_key)
            return '{' + ', '.join((print_rule(k, v) for k, v in items)) + '}'

        def print_dims():
            return self.doprint(expr.shape)
        return 'SparseArray[{}, {}]'.format(print_data(), print_dims())

    def _print_ImmutableDenseNDimArray(self, expr):
        return self.doprint(expr.tolist())

    def _print_ImmutableSparseNDimArray(self, expr):

        def print_string_list(string_list):
            return '{' + ', '.join((a for a in string_list)) + '}'

        def to_mathematica_index(*args):
            """Helper function to change Python style indexing to
            Pathematica indexing.

            Python indexing (0, 1 ... n-1)
            -> Mathematica indexing (1, 2 ... n)
            """
            return tuple((i + 1 for i in args))

        def print_rule(pos, val):
            """Helper function to print a rule of Mathematica"""
            return '{} -> {}'.format(self.doprint(pos), self.doprint(val))

        def print_data():
            """Helper function to print data part of Mathematica
            sparse array.

            It uses the fourth notation ``SparseArray[data,{d1,d2,...}]``
            from
            https://reference.wolfram.com/language/ref/SparseArray.html

            ``data`` must be formatted with rule.
            """
            return print_string_list([print_rule(to_mathematica_index(*expr._get_tuple_index(key)), value) for key, value in sorted(expr._sparse_array.items())])

        def print_dims():
            """Helper function to print dimensions part of Mathematica
            sparse array.

            It uses the fourth notation ``SparseArray[data,{d1,d2,...}]``
            from
            https://reference.wolfram.com/language/ref/SparseArray.html
            """
            return self.doprint(expr.shape)
        return 'SparseArray[{}, {}]'.format(print_data(), print_dims())

    def _print_Function(self, expr):
        if expr.func.__name__ in self.known_functions:
            cond_mfunc = self.known_functions[expr.func.__name__]
            for cond, mfunc in cond_mfunc:
                if cond(*expr.args):
                    return '%s[%s]' % (mfunc, self.stringify(expr.args, ', '))
        elif expr.func.__name__ in self._rewriteable_functions:
            target_f, required_fs = self._rewriteable_functions[expr.func.__name__]
            if self._can_print(target_f) and all((self._can_print(f) for f in required_fs)):
                return self._print(expr.rewrite(target_f))
        return expr.func.__name__ + '[%s]' % self.stringify(expr.args, ', ')
    _print_MinMaxBase = _print_Function

    def _print_LambertW(self, expr):
        if len(expr.args) == 1:
            return 'ProductLog[{}]'.format(self._print(expr.args[0]))
        return 'ProductLog[{}, {}]'.format(self._print(expr.args[1]), self._print(expr.args[0]))

    def _print_Integral(self, expr):
        if len(expr.variables) == 1 and (not expr.limits[0][1:]):
            args = [expr.args[0], expr.variables[0]]
        else:
            args = expr.args
        return 'Hold[Integrate[' + ', '.join((self.doprint(a) for a in args)) + ']]'

    def _print_Sum(self, expr):
        return 'Hold[Sum[' + ', '.join((self.doprint(a) for a in expr.args)) + ']]'

    def _print_Derivative(self, expr):
        dexpr = expr.expr
        dvars = [i[0] if i[1] == 1 else i for i in expr.variable_count]
        return 'Hold[D[' + ', '.join((self.doprint(a) for a in [dexpr] + dvars)) + ']]'

    def _get_comment(self, text):
        return '(* {} *)'.format(text)