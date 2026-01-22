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
def _print_standard_quadratic_NL(self, quadratic_vars, quadratic_coefs):
    OUTPUT = self._OUTPUT
    nary_sum_str, binary_sum_str, coef_term_str = self._op_string[EXPR.SumExpressionBase]
    assert len(quadratic_vars) == len(quadratic_coefs)
    if len(quadratic_vars) == 1:
        pass
    else:
        if len(quadratic_vars) == 2:
            OUTPUT.write(binary_sum_str)
        else:
            assert len(quadratic_vars) > 2
            OUTPUT.write(nary_sum_str % len(quadratic_vars))
        old_quadratic_vars = quadratic_vars
        old_quadratic_coefs = quadratic_coefs
        self_varID_map = self._varID_map
        quadratic_vars = []
        quadratic_coefs = []
        for i, (v1, v2) in sorted(enumerate(old_quadratic_vars), key=lambda x: (self_varID_map[id(x[1][0])], self_varID_map[id(x[1][1])])):
            quadratic_coefs.append(old_quadratic_coefs[i])
            if self_varID_map[id(v1)] <= self_varID_map[id(v2)]:
                quadratic_vars.append((v1, v2))
            else:
                quadratic_vars.append((v2, v1))
    for i in range(len(quadratic_vars)):
        coef = quadratic_coefs[i]
        v1, v2 = quadratic_vars[i]
        if coef != 1:
            OUTPUT.write(coef_term_str % coef)
        self._print_quad_term(v1, v2)