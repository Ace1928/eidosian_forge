from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
class MpmathPrinter(PythonCodePrinter):
    """
    Lambda printer for mpmath which maintains precision for floats
    """
    printmethod = '_mpmathcode'
    language = 'Python with mpmath'
    _kf = dict(chain(_known_functions.items(), [(k, 'mpmath.' + v) for k, v in _known_functions_mpmath.items()]))
    _kc = {k: 'mpmath.' + v for k, v in _known_constants_mpmath.items()}

    def _print_Float(self, e):
        args = str(tuple(map(int, e._mpf_)))
        return '{func}({args})'.format(func=self._module_format('mpmath.mpf'), args=args)

    def _print_Rational(self, e):
        return '{func}({p})/{func}({q})'.format(func=self._module_format('mpmath.mpf'), q=self._print(e.q), p=self._print(e.p))

    def _print_Half(self, e):
        return self._print_Rational(e)

    def _print_uppergamma(self, e):
        return '{}({}, {}, {})'.format(self._module_format('mpmath.gammainc'), self._print(e.args[0]), self._print(e.args[1]), self._module_format('mpmath.inf'))

    def _print_lowergamma(self, e):
        return '{}({}, 0, {})'.format(self._module_format('mpmath.gammainc'), self._print(e.args[0]), self._print(e.args[1]))

    def _print_log2(self, e):
        return '{0}({1})/{0}(2)'.format(self._module_format('mpmath.log'), self._print(e.args[0]))

    def _print_log1p(self, e):
        return '{}({})'.format(self._module_format('mpmath.log1p'), self._print(e.args[0]))

    def _print_Pow(self, expr, rational=False):
        return self._hprint_Pow(expr, rational=rational, sqrt='mpmath.sqrt')

    def _print_Integral(self, e):
        integration_vars, limits = _unpack_integral_limits(e)
        return '{}(lambda {}: {}, {})'.format(self._module_format('mpmath.quad'), ', '.join(map(self._print, integration_vars)), self._print(e.args[0]), ', '.join(('(%s, %s)' % tuple(map(self._print, l)) for l in limits)))