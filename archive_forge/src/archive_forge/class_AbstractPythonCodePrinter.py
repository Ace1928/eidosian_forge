from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter
class AbstractPythonCodePrinter(CodePrinter):
    printmethod = '_pythoncode'
    language = 'Python'
    reserved_words = _kw
    modules = None
    tab = '    '
    _kf = dict(chain(_known_functions.items(), [(k, 'math.' + v) for k, v in _known_functions_math.items()]))
    _kc = {k: 'math.' + v for k, v in _known_constants_math.items()}
    _operators = {'and': 'and', 'or': 'or', 'not': 'not'}
    _default_settings = dict(CodePrinter._default_settings, user_functions={}, precision=17, inline=True, fully_qualified_modules=True, contract=False, standard='python3')

    def __init__(self, settings=None):
        super().__init__(settings)
        std = self._settings['standard']
        if std is None:
            import sys
            std = 'python{}'.format(sys.version_info.major)
        if std != 'python3':
            raise ValueError('Only Python 3 is supported.')
        self.standard = std
        self.module_imports = defaultdict(set)
        self.known_functions = dict(self._kf, **(settings or {}).get('user_functions', {}))
        self.known_constants = dict(self._kc, **(settings or {}).get('user_constants', {}))

    def _declare_number_const(self, name, value):
        return '%s = %s' % (name, value)

    def _module_format(self, fqn, register=True):
        parts = fqn.split('.')
        if register and len(parts) > 1:
            self.module_imports['.'.join(parts[:-1])].add(parts[-1])
        if self._settings['fully_qualified_modules']:
            return fqn
        else:
            return fqn.split('(')[0].split('[')[0].split('.')[-1]

    def _format_code(self, lines):
        return lines

    def _get_statement(self, codestring):
        return '{}'.format(codestring)

    def _get_comment(self, text):
        return '  # {}'.format(text)

    def _expand_fold_binary_op(self, op, args):
        """
        This method expands a fold on binary operations.

        ``functools.reduce`` is an example of a folded operation.

        For example, the expression

        `A + B + C + D`

        is folded into

        `((A + B) + C) + D`
        """
        if len(args) == 1:
            return self._print(args[0])
        else:
            return '%s(%s, %s)' % (self._module_format(op), self._expand_fold_binary_op(op, args[:-1]), self._print(args[-1]))

    def _expand_reduce_binary_op(self, op, args):
        """
        This method expands a reductin on binary operations.

        Notice: this is NOT the same as ``functools.reduce``.

        For example, the expression

        `A + B + C + D`

        is reduced into:

        `(A + B) + (C + D)`
        """
        if len(args) == 1:
            return self._print(args[0])
        else:
            N = len(args)
            Nhalf = N // 2
            return '%s(%s, %s)' % (self._module_format(op), self._expand_reduce_binary_op(args[:Nhalf]), self._expand_reduce_binary_op(args[Nhalf:]))

    def _print_NaN(self, expr):
        return "float('nan')"

    def _print_Infinity(self, expr):
        return "float('inf')"

    def _print_NegativeInfinity(self, expr):
        return "float('-inf')"

    def _print_ComplexInfinity(self, expr):
        return self._print_NaN(expr)

    def _print_Mod(self, expr):
        PREC = precedence(expr)
        return '{} % {}'.format(*(self.parenthesize(x, PREC) for x in expr.args))

    def _print_Piecewise(self, expr):
        result = []
        i = 0
        for arg in expr.args:
            e = arg.expr
            c = arg.cond
            if i == 0:
                result.append('(')
            result.append('(')
            result.append(self._print(e))
            result.append(')')
            result.append(' if ')
            result.append(self._print(c))
            result.append(' else ')
            i += 1
        result = result[:-1]
        if result[-1] == 'True':
            result = result[:-2]
            result.append(')')
        else:
            result.append(' else None)')
        return ''.join(result)

    def _print_Relational(self, expr):
        """Relational printer for Equality and Unequality"""
        op = {'==': 'equal', '!=': 'not_equal', '<': 'less', '<=': 'less_equal', '>': 'greater', '>=': 'greater_equal'}
        if expr.rel_op in op:
            lhs = self._print(expr.lhs)
            rhs = self._print(expr.rhs)
            return '({lhs} {op} {rhs})'.format(op=expr.rel_op, lhs=lhs, rhs=rhs)
        return super()._print_Relational(expr)

    def _print_ITE(self, expr):
        from sympy.functions.elementary.piecewise import Piecewise
        return self._print(expr.rewrite(Piecewise))

    def _print_Sum(self, expr):
        loops = ('for {i} in range({a}, {b}+1)'.format(i=self._print(i), a=self._print(a), b=self._print(b)) for i, a, b in expr.limits)
        return '(builtins.sum({function} {loops}))'.format(function=self._print(expr.function), loops=' '.join(loops))

    def _print_ImaginaryUnit(self, expr):
        return '1j'

    def _print_KroneckerDelta(self, expr):
        a, b = expr.args
        return '(1 if {a} == {b} else 0)'.format(a=self._print(a), b=self._print(b))

    def _print_MatrixBase(self, expr):
        name = expr.__class__.__name__
        func = self.known_functions.get(name, name)
        return '%s(%s)' % (func, self._print(expr.tolist()))
    _print_SparseRepMatrix = _print_MutableSparseMatrix = _print_ImmutableSparseMatrix = _print_Matrix = _print_DenseMatrix = _print_MutableDenseMatrix = _print_ImmutableMatrix = _print_ImmutableDenseMatrix = lambda self, expr: self._print_MatrixBase(expr)

    def _indent_codestring(self, codestring):
        return '\n'.join([self.tab + line for line in codestring.split('\n')])

    def _print_FunctionDefinition(self, fd):
        body = '\n'.join((self._print(arg) for arg in fd.body))
        return 'def {name}({parameters}):\n{body}'.format(name=self._print(fd.name), parameters=', '.join([self._print(var.symbol) for var in fd.parameters]), body=self._indent_codestring(body))

    def _print_While(self, whl):
        body = '\n'.join((self._print(arg) for arg in whl.body))
        return 'while {cond}:\n{body}'.format(cond=self._print(whl.condition), body=self._indent_codestring(body))

    def _print_Declaration(self, decl):
        return '%s = %s' % (self._print(decl.variable.symbol), self._print(decl.variable.value))

    def _print_Return(self, ret):
        arg, = ret.args
        return 'return %s' % self._print(arg)

    def _print_Print(self, prnt):
        print_args = ', '.join((self._print(arg) for arg in prnt.print_args))
        if prnt.format_string != None:
            print_args = '{} % ({})'.format(self._print(prnt.format_string), print_args)
        if prnt.file != None:
            print_args += ', file=%s' % self._print(prnt.file)
        return 'print(%s)' % print_args

    def _print_Stream(self, strm):
        if str(strm.name) == 'stdout':
            return self._module_format('sys.stdout')
        elif str(strm.name) == 'stderr':
            return self._module_format('sys.stderr')
        else:
            return self._print(strm.name)

    def _print_NoneToken(self, arg):
        return 'None'

    def _hprint_Pow(self, expr, rational=False, sqrt='math.sqrt'):
        """Printing helper function for ``Pow``

        Notes
        =====

        This preprocesses the ``sqrt`` as math formatter and prints division

        Examples
        ========

        >>> from sympy import sqrt
        >>> from sympy.printing.pycode import PythonCodePrinter
        >>> from sympy.abc import x

        Python code printer automatically looks up ``math.sqrt``.

        >>> printer = PythonCodePrinter()
        >>> printer._hprint_Pow(sqrt(x), rational=True)
        'x**(1/2)'
        >>> printer._hprint_Pow(sqrt(x), rational=False)
        'math.sqrt(x)'
        >>> printer._hprint_Pow(1/sqrt(x), rational=True)
        'x**(-1/2)'
        >>> printer._hprint_Pow(1/sqrt(x), rational=False)
        '1/math.sqrt(x)'
        >>> printer._hprint_Pow(1/x, rational=False)
        '1/x'
        >>> printer._hprint_Pow(1/x, rational=True)
        'x**(-1)'

        Using sqrt from numpy or mpmath

        >>> printer._hprint_Pow(sqrt(x), sqrt='numpy.sqrt')
        'numpy.sqrt(x)'
        >>> printer._hprint_Pow(sqrt(x), sqrt='mpmath.sqrt')
        'mpmath.sqrt(x)'

        See Also
        ========

        sympy.printing.str.StrPrinter._print_Pow
        """
        PREC = precedence(expr)
        if expr.exp == S.Half and (not rational):
            func = self._module_format(sqrt)
            arg = self._print(expr.base)
            return '{func}({arg})'.format(func=func, arg=arg)
        if expr.is_commutative and (not rational):
            if -expr.exp is S.Half:
                func = self._module_format(sqrt)
                num = self._print(S.One)
                arg = self._print(expr.base)
                return f'{num}/{func}({arg})'
            if expr.exp is S.NegativeOne:
                num = self._print(S.One)
                arg = self.parenthesize(expr.base, PREC, strict=False)
                return f'{num}/{arg}'
        base_str = self.parenthesize(expr.base, PREC, strict=False)
        exp_str = self.parenthesize(expr.exp, PREC, strict=False)
        return '{}**{}'.format(base_str, exp_str)