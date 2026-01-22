import itertools
import logging
import operator
import os
import time
from math import isclose
from pyomo.common.fileutils import find_library
from pyomo.common.gc_manager import PauseGC
from pyomo.opt import ProblemFormat, AbstractProblemWriter, WriterFactory
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import (
from pyomo.core.base import (
import pyomo.core.base.suffix
from pyomo.repn.standard_repn import generate_standard_repn
import pyomo.core.kernel.suffix
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.expression import IIdentityExpression
from pyomo.core.kernel.variable import IVariable
def _print_nonlinear_terms_NL(self, exp):
    OUTPUT = self._OUTPUT
    exp_type = type(exp)
    if exp_type is list:
        nary_sum_str, binary_sum_str, coef_term_str = self._op_string[EXPR.SumExpressionBase]
        n = len(exp)
        if n > 2:
            OUTPUT.write(nary_sum_str % n)
            for i in range(0, n):
                assert exp[i].__class__ is tuple
                coef = exp[i][0]
                child_exp = exp[i][1]
                if coef != 1:
                    OUTPUT.write(coef_term_str % coef)
                self._print_nonlinear_terms_NL(child_exp)
        else:
            for i in range(0, n):
                assert exp[i].__class__ is tuple
                coef = exp[i][0]
                child_exp = exp[i][1]
                if i != n - 1:
                    OUTPUT.write(binary_sum_str)
                if coef != 1:
                    OUTPUT.write(coef_term_str % coef)
                self._print_nonlinear_terms_NL(child_exp)
    elif exp_type in native_numeric_types:
        OUTPUT.write(self._op_string[NumericConstant] % exp)
    elif exp.is_expression_type():
        if not exp.is_potentially_variable():
            OUTPUT.write(self._op_string[NumericConstant] % value(exp))
        elif exp.__class__ is EXPR.SumExpression or exp.__class__ is EXPR.LinearExpression:
            nary_sum_str, binary_sum_str, coef_term_str = self._op_string[EXPR.SumExpressionBase]
            n = exp.nargs()
            const = 0
            vargs = []
            for v in exp.args:
                if v.__class__ in native_numeric_types:
                    const += v
                else:
                    vargs.append(v)
            if not isclose(const, 0.0):
                vargs.append(const)
            n = len(vargs)
            if n == 2:
                OUTPUT.write(binary_sum_str)
                self._print_nonlinear_terms_NL(vargs[0])
                self._print_nonlinear_terms_NL(vargs[1])
            elif n == 1:
                self._print_nonlinear_terms_NL(vargs[0])
            else:
                OUTPUT.write(nary_sum_str % n)
                for child_exp in vargs:
                    self._print_nonlinear_terms_NL(child_exp)
        elif exp_type is EXPR.SumExpressionBase:
            nary_sum_str, binary_sum_str, coef_term_str = self._op_string[EXPR.SumExpressionBase]
            OUTPUT.write(binary_sum_str)
            self._print_nonlinear_terms_NL(exp.arg(0))
            self._print_nonlinear_terms_NL(exp.arg(1))
        elif exp_type is EXPR.MonomialTermExpression:
            prod_str = self._op_string[EXPR.ProductExpression]
            OUTPUT.write(prod_str)
            self._print_nonlinear_terms_NL(value(exp.arg(0)))
            self._print_nonlinear_terms_NL(exp.arg(1))
        elif exp_type is EXPR.ProductExpression:
            prod_str = self._op_string[EXPR.ProductExpression]
            OUTPUT.write(prod_str)
            self._print_nonlinear_terms_NL(exp.arg(0))
            self._print_nonlinear_terms_NL(exp.arg(1))
        elif exp_type is EXPR.DivisionExpression:
            assert exp.nargs() == 2
            div_str = self._op_string[EXPR.DivisionExpression]
            OUTPUT.write(div_str)
            self._print_nonlinear_terms_NL(exp.arg(0))
            self._print_nonlinear_terms_NL(exp.arg(1))
        elif exp_type is EXPR.NegationExpression:
            assert exp.nargs() == 1
            OUTPUT.write(self._op_string[EXPR.NegationExpression])
            self._print_nonlinear_terms_NL(exp.arg(0))
        elif exp_type is EXPR.ExternalFunctionExpression:
            if exp.is_fixed():
                self._print_nonlinear_terms_NL(exp())
                return
            fun_str, string_arg_str = self._op_string[EXPR.ExternalFunctionExpression]
            if not self._symbolic_solver_labels:
                OUTPUT.write(fun_str % (self.external_byFcn[exp._fcn._function][1], exp.nargs()))
            else:
                OUTPUT.write(fun_str % (self.external_byFcn[exp._fcn._function][1], exp.nargs(), exp.name))
            for arg in exp.args:
                if isinstance(arg, str):
                    OUTPUT.flush()
                    with os.fdopen(OUTPUT.fileno(), mode='w+', closefd=False, newline='\n') as TMP:
                        TMP.write(string_arg_str % (len(arg), arg))
                elif type(arg) in native_numeric_types:
                    self._print_nonlinear_terms_NL(arg)
                elif arg.is_fixed():
                    self._print_nonlinear_terms_NL(arg())
                else:
                    self._print_nonlinear_terms_NL(arg)
        elif exp_type is EXPR.PowExpression:
            intr_expr_str = self._op_string['pow']
            OUTPUT.write(intr_expr_str)
            self._print_nonlinear_terms_NL(exp.arg(0))
            self._print_nonlinear_terms_NL(exp.arg(1))
        elif isinstance(exp, EXPR.UnaryFunctionExpression):
            assert exp.nargs() == 1
            intr_expr_str = self._op_string.get(exp.name)
            if intr_expr_str is not None:
                OUTPUT.write(intr_expr_str)
            else:
                logger.error('Unsupported unary function ({0})'.format(exp.name))
                raise TypeError("ASL writer does not support '%s' expressions" % exp.name)
            self._print_nonlinear_terms_NL(exp.arg(0))
        elif exp_type is EXPR.Expr_ifExpression:
            OUTPUT.write(self._op_string[EXPR.Expr_ifExpression])
            for arg in exp.args:
                self._print_nonlinear_terms_NL(arg)
        elif exp_type is EXPR.InequalityExpression:
            and_str, lt_str, le_str = self._op_string[EXPR.InequalityExpression]
            left = exp.arg(0)
            right = exp.arg(1)
            if exp._strict:
                OUTPUT.write(lt_str)
            else:
                OUTPUT.write(le_str)
            self._print_nonlinear_terms_NL(left)
            self._print_nonlinear_terms_NL(right)
        elif exp_type is EXPR.RangedExpression:
            and_str, lt_str, le_str = self._op_string[EXPR.InequalityExpression]
            left = exp.arg(0)
            middle = exp.arg(1)
            right = exp.arg(2)
            OUTPUT.write(and_str)
            if exp._strict[0]:
                OUTPUT.write(lt_str)
            else:
                OUTPUT.write(le_str)
            self._print_nonlinear_terms_NL(left)
            self._print_nonlinear_terms_NL(middle)
            if exp._strict[1]:
                OUTPUT.write(lt_str)
            else:
                OUTPUT.write(le_str)
            self._print_nonlinear_terms_NL(middle)
            self._print_nonlinear_terms_NL(right)
        elif exp_type is EXPR.EqualityExpression:
            OUTPUT.write(self._op_string[EXPR.EqualityExpression])
            self._print_nonlinear_terms_NL(exp.arg(0))
            self._print_nonlinear_terms_NL(exp.arg(1))
        elif isinstance(exp, (_ExpressionData, IIdentityExpression)):
            self._print_nonlinear_terms_NL(exp.expr)
        else:
            raise ValueError('Unsupported expression type (%s) in _print_nonlinear_terms_NL' % exp_type)
    elif isinstance(exp, (var._VarData, IVariable)) and (not exp.is_fixed()):
        if not self._symbolic_solver_labels:
            OUTPUT.write(self._op_string[var._VarData] % self.ampl_var_id[self._varID_map[id(exp)]])
        else:
            OUTPUT.write(self._op_string[var._VarData] % (self.ampl_var_id[self._varID_map[id(exp)]], self._name_labeler(exp)))
    elif isinstance(exp, param._ParamData):
        OUTPUT.write(self._op_string[param._ParamData] % value(exp))
    elif isinstance(exp, NumericConstant) or exp.is_fixed():
        OUTPUT.write(self._op_string[NumericConstant] % value(exp))
    else:
        raise ValueError('Unsupported expression type (%s) in _print_nonlinear_terms_NL' % exp_type)