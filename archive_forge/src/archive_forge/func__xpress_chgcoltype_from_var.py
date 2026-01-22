from pyomo.core.base.PyomoModel import ConcreteModel
from pyomo.solvers.plugins.solvers.xpress_direct import XpressDirect
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.expr.numvalue import value, is_fixed
import pyomo.core.expr as EXPR
from pyomo.opt.base import SolverFactory
import collections
def _xpress_chgcoltype_from_var(self, var):
    """
        This function takes a pyomo variable and returns the appropriate xpress variable type
        for use in xpress.problem.chgcoltype
        :param var: pyomo.core.base.var.Var
        :return: xpress.continuous or xpress.binary or xpress.integer
        """
    if var.is_binary():
        vartype = 'B'
    elif var.is_integer():
        vartype = 'I'
    elif var.is_continuous():
        vartype = 'C'
    else:
        raise ValueError('Variable domain type is not recognized for {0}'.format(var.domain))
    return vartype