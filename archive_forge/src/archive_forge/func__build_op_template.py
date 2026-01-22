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
def _build_op_template():
    _op_template = {}
    _op_comment = {}
    prod_template = 'o2{C}\n'
    prod_comment = '\t#*'
    div_template = 'o3{C}\n'
    div_comment = '\t#/'
    _op_template[EXPR.ProductExpression] = prod_template
    _op_comment[EXPR.ProductExpression] = prod_comment
    _op_template[EXPR.DivisionExpression] = div_template
    _op_comment[EXPR.DivisionExpression] = div_comment
    _op_template[EXPR.ExternalFunctionExpression] = ('f%d %d{C}\n', 'h%d:%s{C}\n')
    _op_comment[EXPR.ExternalFunctionExpression] = ('\t#%s', '')
    for opname in _intrinsic_function_operators:
        _op_template[opname] = _intrinsic_function_operators[opname] + '{C}\n'
        _op_comment[opname] = '\t#' + opname
    _op_template[EXPR.Expr_ifExpression] = 'o35{C}\n'
    _op_comment[EXPR.Expr_ifExpression] = '\t#if'
    _op_template[EXPR.InequalityExpression] = ('o21{C}\n', 'o22{C}\n', 'o23{C}\n')
    _op_comment[EXPR.InequalityExpression] = ('\t#and', '\t#lt', '\t#le')
    _op_template[EXPR.EqualityExpression] = 'o24{C}\n'
    _op_comment[EXPR.EqualityExpression] = '\t#eq'
    _op_template[var._VarData] = 'v%d{C}\n'
    _op_comment[var._VarData] = '\t#%s'
    _op_template[param._ParamData] = 'n%r{C}\n'
    _op_comment[param._ParamData] = ''
    _op_template[NumericConstant] = 'n%r{C}\n'
    _op_comment[NumericConstant] = ''
    _op_template[EXPR.SumExpressionBase] = ('o54{C}\n%d\n', 'o0{C}\n', 'o2\n' + _op_template[NumericConstant])
    _op_comment[EXPR.SumExpressionBase] = ('\t#sumlist', '\t#+', _op_comment[NumericConstant])
    _op_template[EXPR.NegationExpression] = 'o16{C}\n'
    _op_comment[EXPR.NegationExpression] = '\t#-'
    return (_op_template, _op_comment)