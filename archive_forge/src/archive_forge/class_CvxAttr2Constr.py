import numpy as np
import scipy.sparse as sp
from cvxpy.atoms import diag, reshape
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable, upper_tri_to_full
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution
class CvxAttr2Constr(Reduction):
    """Expand convex variable attributes into constraints."""

    def accepts(self, problem) -> bool:
        return True

    def apply(self, problem):
        if not attributes_present(problem.variables(), CONVEX_ATTRIBUTES):
            return (problem, ())
        id2new_var = {}
        id2new_obj = {}
        id2old_var = {}
        constr = []
        for var in problem.variables():
            if var.id not in id2new_var:
                id2old_var[var.id] = var
                new_var = False
                new_attr = var.attributes.copy()
                for key in CONVEX_ATTRIBUTES:
                    if new_attr[key]:
                        new_var = True
                        new_attr[key] = False
                if attributes_present([var], SYMMETRIC_ATTRIBUTES):
                    n = var.shape[0]
                    shape = (n * (n + 1) // 2, 1)
                    upper_tri = Variable(shape, var_id=var.id, **new_attr)
                    upper_tri.set_variable_of_provenance(var)
                    id2new_var[var.id] = upper_tri
                    fill_coeff = Constant(upper_tri_to_full(n))
                    full_mat = fill_coeff @ upper_tri
                    obj = reshape(full_mat, (n, n))
                elif var.attributes['diag']:
                    diag_var = Variable(var.shape[0], var_id=var.id, **new_attr)
                    diag_var.set_variable_of_provenance(var)
                    id2new_var[var.id] = diag_var
                    obj = diag(diag_var)
                elif new_var:
                    obj = Variable(var.shape, var_id=var.id, **new_attr)
                    obj.set_variable_of_provenance(var)
                    id2new_var[var.id] = obj
                else:
                    obj = var
                    id2new_var[var.id] = obj
                id2new_obj[id(var)] = obj
                if var.is_pos() or var.is_nonneg():
                    constr.append(obj >= 0)
                elif var.is_neg() or var.is_nonpos():
                    constr.append(obj <= 0)
                elif var.is_psd():
                    constr.append(obj >> 0)
                elif var.attributes['NSD']:
                    constr.append(obj << 0)
        obj = problem.objective.tree_copy(id_objects=id2new_obj)
        cons_id_map = {}
        for cons in problem.constraints:
            constr.append(cons.tree_copy(id_objects=id2new_obj))
            cons_id_map[cons.id] = constr[-1].id
        inverse_data = (id2new_var, id2old_var, cons_id_map)
        return (cvxtypes.problem()(obj, constr), inverse_data)

    def invert(self, solution, inverse_data):
        if not inverse_data:
            return solution
        id2new_var, id2old_var, cons_id_map = inverse_data
        pvars = {}
        for id, var in id2old_var.items():
            new_var = id2new_var[id]
            if new_var.id in solution.primal_vars:
                pvars[id] = recover_value_for_variable(var, solution.primal_vars[new_var.id])
        dvars = {orig_id: solution.dual_vars[vid] for orig_id, vid in cons_id_map.items() if vid in solution.dual_vars}
        return Solution(solution.status, solution.opt_val, pvars, dvars, solution.attr)