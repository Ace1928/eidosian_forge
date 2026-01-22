from collections import defaultdict
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor
import pyomo.core.expr.numeric_expr as numeric_expr
from pyomo.core.expr.visitor import (
from pyomo.core.expr.numvalue import nonpyomo_leaf_types, value
from pyomo.core.expr.numvalue import is_fixed
import pyomo.contrib.fbbt.interval as interval
import math
from pyomo.core.base.block import Block
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.gdp import Disjunct
from pyomo.core.base.expression import _GeneralExpressionData, ScalarExpression
import logging
from pyomo.common.errors import InfeasibleConstraintException, PyomoException
from pyomo.common.config import (
from pyomo.common.numeric_types import native_types
from the constraint, we know that 1 <= x*y + z <= 1, so we may 
def _fbbt_block(m, config):
    """
    Feasibility based bounds tightening (FBBT) for a block or model. This
    loops through all of the constraints in the block and performs
    FBBT on each constraint (see the docstring for _fbbt_con()).
    Through this processes, any variables whose bounds improve
    by more than tol are collected, and FBBT is
    performed again on all constraints involving those variables.
    This process is continued until no variable bounds are improved
    by more than tol.

    Parameters
    ----------
    m: pyomo.core.base.block.Block or pyomo.core.base.PyomoModel.ConcreteModel
    config: ConfigDict
        See the docs for fbbt

    Returns
    -------
    new_var_bounds: ComponentMap
        A ComponentMap mapping from variables a tuple containing the lower and upper bounds, respectively, computed
        from FBBT.
    """
    new_var_bounds = ComponentMap()
    var_to_con_map = ComponentMap()
    var_lbs = ComponentMap()
    var_ubs = ComponentMap()
    n_cons = 0
    for c in m.component_data_objects(ctype=Constraint, active=True, descend_into=config.descend_into, sort=True):
        for v in identify_variables(c.body):
            if v not in var_to_con_map:
                var_to_con_map[v] = list()
            if v.lb is None:
                var_lbs[v] = -interval.inf
            else:
                var_lbs[v] = v.lb
            if v.ub is None:
                var_ubs[v] = interval.inf
            else:
                var_ubs[v] = v.ub
            var_to_con_map[v].append(c)
        n_cons += 1
    for _v in m.component_data_objects(ctype=Var, active=True, descend_into=True, sort=True):
        if _v.is_fixed():
            _v.setlb(_v.value)
            _v.setub(_v.value)
            new_var_bounds[_v] = (_v.value, _v.value)
    n_fbbt = 0
    improved_vars = ComponentSet()
    for c in m.component_data_objects(ctype=Constraint, active=True, descend_into=config.descend_into, sort=True):
        _new_var_bounds = _fbbt_con(c, config)
        n_fbbt += 1
        new_var_bounds.update(_new_var_bounds)
        for v, bnds in _new_var_bounds.items():
            vlb, vub = bnds
            if vlb is not None:
                if vlb > var_lbs[v] + config.improvement_tol:
                    improved_vars.add(v)
                    var_lbs[v] = vlb
            if vub is not None:
                if vub < var_ubs[v] - config.improvement_tol:
                    improved_vars.add(v)
                    var_ubs[v] = vub
    while len(improved_vars) > 0:
        if n_fbbt >= n_cons * config.max_iter:
            break
        v = improved_vars.pop()
        for c in var_to_con_map[v]:
            _new_var_bounds = _fbbt_con(c, config)
            n_fbbt += 1
            new_var_bounds.update(_new_var_bounds)
            for _v, bnds in _new_var_bounds.items():
                _vlb, _vub = bnds
                if _vlb is not None:
                    if _vlb > var_lbs[_v] + config.improvement_tol:
                        improved_vars.add(_v)
                        var_lbs[_v] = _vlb
                if _vub is not None:
                    if _vub < var_ubs[_v] - config.improvement_tol:
                        improved_vars.add(_v)
                        var_ubs[_v] = _vub
    return new_var_bounds