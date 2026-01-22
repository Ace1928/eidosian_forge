from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
class FuncDeclRef(AstRef):
    """Function declaration. Every constant and function have an associated declaration.

    The declaration assigns a name, a sort (i.e., type), and for function
    the sort (i.e., type) of each of its arguments. Note that, in Z3,
    a constant is a function with 0 arguments.
    """

    def as_ast(self):
        return Z3_func_decl_to_ast(self.ctx_ref(), self.ast)

    def get_id(self):
        return Z3_get_ast_id(self.ctx_ref(), self.as_ast())

    def as_func_decl(self):
        return self.ast

    def name(self):
        """Return the name of the function declaration `self`.

        >>> f = Function('f', IntSort(), IntSort())
        >>> f.name()
        'f'
        >>> isinstance(f.name(), str)
        True
        """
        return _symbol2py(self.ctx, Z3_get_decl_name(self.ctx_ref(), self.ast))

    def arity(self):
        """Return the number of arguments of a function declaration.
        If `self` is a constant, then `self.arity()` is 0.

        >>> f = Function('f', IntSort(), RealSort(), BoolSort())
        >>> f.arity()
        2
        """
        return int(Z3_get_arity(self.ctx_ref(), self.ast))

    def domain(self, i):
        """Return the sort of the argument `i` of a function declaration.
        This method assumes that `0 <= i < self.arity()`.

        >>> f = Function('f', IntSort(), RealSort(), BoolSort())
        >>> f.domain(0)
        Int
        >>> f.domain(1)
        Real
        """
        return _to_sort_ref(Z3_get_domain(self.ctx_ref(), self.ast, i), self.ctx)

    def range(self):
        """Return the sort of the range of a function declaration.
        For constants, this is the sort of the constant.

        >>> f = Function('f', IntSort(), RealSort(), BoolSort())
        >>> f.range()
        Bool
        """
        return _to_sort_ref(Z3_get_range(self.ctx_ref(), self.ast), self.ctx)

    def kind(self):
        """Return the internal kind of a function declaration.
        It can be used to identify Z3 built-in functions such as addition, multiplication, etc.

        >>> x = Int('x')
        >>> d = (x + 1).decl()
        >>> d.kind() == Z3_OP_ADD
        True
        >>> d.kind() == Z3_OP_MUL
        False
        """
        return Z3_get_decl_kind(self.ctx_ref(), self.ast)

    def params(self):
        ctx = self.ctx
        n = Z3_get_decl_num_parameters(self.ctx_ref(), self.ast)
        result = [None for i in range(n)]
        for i in range(n):
            k = Z3_get_decl_parameter_kind(self.ctx_ref(), self.ast, i)
            if k == Z3_PARAMETER_INT:
                result[i] = Z3_get_decl_int_parameter(self.ctx_ref(), self.ast, i)
            elif k == Z3_PARAMETER_DOUBLE:
                result[i] = Z3_get_decl_double_parameter(self.ctx_ref(), self.ast, i)
            elif k == Z3_PARAMETER_RATIONAL:
                result[i] = Z3_get_decl_rational_parameter(self.ctx_ref(), self.ast, i)
            elif k == Z3_PARAMETER_SYMBOL:
                result[i] = Z3_get_decl_symbol_parameter(self.ctx_ref(), self.ast, i)
            elif k == Z3_PARAMETER_SORT:
                result[i] = SortRef(Z3_get_decl_sort_parameter(self.ctx_ref(), self.ast, i), ctx)
            elif k == Z3_PARAMETER_AST:
                result[i] = ExprRef(Z3_get_decl_ast_parameter(self.ctx_ref(), self.ast, i), ctx)
            elif k == Z3_PARAMETER_FUNC_DECL:
                result[i] = FuncDeclRef(Z3_get_decl_func_decl_parameter(self.ctx_ref(), self.ast, i), ctx)
            else:
                assert False
        return result

    def __call__(self, *args):
        """Create a Z3 application expression using the function `self`, and the given arguments.

        The arguments must be Z3 expressions. This method assumes that
        the sorts of the elements in `args` match the sorts of the
        domain. Limited coercion is supported.  For example, if
        args[0] is a Python integer, and the function expects a Z3
        integer, then the argument is automatically converted into a
        Z3 integer.

        >>> f = Function('f', IntSort(), RealSort(), BoolSort())
        >>> x = Int('x')
        >>> y = Real('y')
        >>> f(x, y)
        f(x, y)
        >>> f(x, x)
        f(x, ToReal(x))
        """
        args = _get_args(args)
        num = len(args)
        _args = (Ast * num)()
        saved = []
        for i in range(num):
            tmp = self.domain(i).cast(args[i])
            saved.append(tmp)
            _args[i] = tmp.as_ast()
        return _to_expr_ref(Z3_mk_app(self.ctx_ref(), self.ast, len(args), _args), self.ctx)