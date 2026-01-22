from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock
from .core import glpk_path, operating_system, log
import os
from .. import constants
class PYGLPK(LpSolver):
    """
    The glpk LP/MIP solver (via its python interface)

    Copyright Christophe-Marie Duquesne 2012

    The glpk variables are available (after a solve) in var.solverVar
    The glpk constraints are available in constraint.solverConstraint
    The Model is in prob.solverModel
    """
    name = 'PYGLPK'
    try:
        global glpk
        import glpk.glpkpi as glpk
    except:

        def available(self):
            """True if the solver is available"""
            return False

        def actualSolve(self, lp, callback=None):
            """Solve a well formulated lp problem"""
            raise PulpSolverError('GLPK: Not Available')
    else:

        def __init__(self, mip=True, msg=True, timeLimit=None, gapRel=None, **solverParams):
            """
            Initializes the glpk solver.

            @param mip: if False the solver will solve a MIP as an LP
            @param msg: displays information from the solver to stdout
            @param timeLimit: not handled
            @param gapRel: not handled
            @param solverParams: not handled
            """
            LpSolver.__init__(self, mip, msg)
            if not self.msg:
                glpk.glp_term_out(glpk.GLP_OFF)

        def findSolutionValues(self, lp):
            prob = lp.solverModel
            if self.mip and self.hasMIPConstraints(lp.solverModel):
                solutionStatus = glpk.glp_mip_status(prob)
            else:
                solutionStatus = glpk.glp_get_status(prob)
            glpkLpStatus = {glpk.GLP_OPT: constants.LpStatusOptimal, glpk.GLP_UNDEF: constants.LpStatusUndefined, glpk.GLP_FEAS: constants.LpStatusOptimal, glpk.GLP_INFEAS: constants.LpStatusInfeasible, glpk.GLP_NOFEAS: constants.LpStatusInfeasible, glpk.GLP_UNBND: constants.LpStatusUnbounded}
            for var in lp.variables():
                if self.mip and self.hasMIPConstraints(lp.solverModel):
                    var.varValue = glpk.glp_mip_col_val(prob, var.glpk_index)
                else:
                    var.varValue = glpk.glp_get_col_prim(prob, var.glpk_index)
                var.dj = glpk.glp_get_col_dual(prob, var.glpk_index)
            for constr in lp.constraints.values():
                if self.mip and self.hasMIPConstraints(lp.solverModel):
                    row_val = glpk.glp_mip_row_val(prob, constr.glpk_index)
                else:
                    row_val = glpk.glp_get_row_prim(prob, constr.glpk_index)
                constr.slack = -constr.constant - row_val
                constr.pi = glpk.glp_get_row_dual(prob, constr.glpk_index)
            lp.resolveOK = True
            for var in lp.variables():
                var.isModified = False
            status = glpkLpStatus.get(solutionStatus, constants.LpStatusUndefined)
            lp.assignStatus(status)
            return status

        def available(self):
            """True if the solver is available"""
            return True

        def hasMIPConstraints(self, solverModel):
            return glpk.glp_get_num_int(solverModel) > 0 or glpk.glp_get_num_bin(solverModel) > 0

        def callSolver(self, lp, callback=None):
            """Solves the problem with glpk"""
            self.solveTime = -clock()
            glpk.glp_adv_basis(lp.solverModel, 0)
            glpk.glp_simplex(lp.solverModel, None)
            if self.mip and self.hasMIPConstraints(lp.solverModel):
                status = glpk.glp_get_status(lp.solverModel)
                if status in (glpk.GLP_OPT, glpk.GLP_UNDEF, glpk.GLP_FEAS):
                    glpk.glp_intopt(lp.solverModel, None)
            self.solveTime += clock()

        def buildSolverModel(self, lp):
            """
            Takes the pulp lp model and translates it into a glpk model
            """
            log.debug('create the glpk model')
            prob = glpk.glp_create_prob()
            glpk.glp_set_prob_name(prob, lp.name)
            log.debug('set the sense of the problem')
            if lp.sense == constants.LpMaximize:
                glpk.glp_set_obj_dir(prob, glpk.GLP_MAX)
            log.debug('add the constraints to the problem')
            glpk.glp_add_rows(prob, len(list(lp.constraints.keys())))
            for i, v in enumerate(lp.constraints.items(), start=1):
                name, constraint = v
                glpk.glp_set_row_name(prob, i, name)
                if constraint.sense == constants.LpConstraintLE:
                    glpk.glp_set_row_bnds(prob, i, glpk.GLP_UP, 0.0, -constraint.constant)
                elif constraint.sense == constants.LpConstraintGE:
                    glpk.glp_set_row_bnds(prob, i, glpk.GLP_LO, -constraint.constant, 0.0)
                elif constraint.sense == constants.LpConstraintEQ:
                    glpk.glp_set_row_bnds(prob, i, glpk.GLP_FX, -constraint.constant, -constraint.constant)
                else:
                    raise PulpSolverError('Detected an invalid constraint type')
                constraint.glpk_index = i
            log.debug('add the variables to the problem')
            glpk.glp_add_cols(prob, len(lp.variables()))
            for j, var in enumerate(lp.variables(), start=1):
                glpk.glp_set_col_name(prob, j, var.name)
                lb = 0.0
                ub = 0.0
                t = glpk.GLP_FR
                if not var.lowBound is None:
                    lb = var.lowBound
                    t = glpk.GLP_LO
                if not var.upBound is None:
                    ub = var.upBound
                    t = glpk.GLP_UP
                if not var.upBound is None and (not var.lowBound is None):
                    if ub == lb:
                        t = glpk.GLP_FX
                    else:
                        t = glpk.GLP_DB
                glpk.glp_set_col_bnds(prob, j, t, lb, ub)
                if var.cat == constants.LpInteger:
                    glpk.glp_set_col_kind(prob, j, glpk.GLP_IV)
                    assert glpk.glp_get_col_kind(prob, j) == glpk.GLP_IV
                var.glpk_index = j
            log.debug('set the objective function')
            for var in lp.variables():
                value = lp.objective.get(var)
                if value:
                    glpk.glp_set_obj_coef(prob, var.glpk_index, value)
            log.debug('set the problem matrix')
            for constraint in lp.constraints.values():
                l = len(list(constraint.items()))
                ind = glpk.intArray(l + 1)
                val = glpk.doubleArray(l + 1)
                for j, v in enumerate(constraint.items(), start=1):
                    var, value = v
                    ind[j] = var.glpk_index
                    val[j] = value
                glpk.glp_set_mat_row(prob, constraint.glpk_index, l, ind, val)
            lp.solverModel = prob

        def actualSolve(self, lp, callback=None):
            """
            Solve a well formulated lp problem

            creates a glpk model, variables and constraints and attaches
            them to the lp model which it then solves
            """
            self.buildSolverModel(lp)
            log.debug('Solve the Model using glpk')
            self.callSolver(lp, callback=callback)
            solutionStatus = self.findSolutionValues(lp)
            for var in lp.variables():
                var.modified = False
            for constraint in lp.constraints.values():
                constraint.modified = False
            return solutionStatus

        def actualResolve(self, lp, callback=None):
            """
            Solve a well formulated lp problem

            uses the old solver and modifies the rhs of the modified
            constraints
            """
            prob = lp.solverModel
            log.debug('Resolve the Model using glpk')
            for constraint in lp.constraints.values():
                i = constraint.glpk_index
                if constraint.modified:
                    if constraint.sense == constants.LpConstraintLE:
                        glpk.glp_set_row_bnds(prob, i, glpk.GLP_UP, 0.0, -constraint.constant)
                    elif constraint.sense == constants.LpConstraintGE:
                        glpk.glp_set_row_bnds(prob, i, glpk.GLP_LO, -constraint.constant, 0.0)
                    elif constraint.sense == constants.LpConstraintEQ:
                        glpk.glp_set_row_bnds(prob, i, glpk.GLP_FX, -constraint.constant, -constraint.constant)
                    else:
                        raise PulpSolverError('Detected an invalid constraint type')
            self.callSolver(lp, callback=callback)
            solutionStatus = self.findSolutionValues(lp)
            for var in lp.variables():
                var.modified = False
            for constraint in lp.constraints.values():
                constraint.modified = False
            return solutionStatus