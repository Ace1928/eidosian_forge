import numpy as np
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.versioning import Version
class CLARABEL(ConicSolver):
    """An interface for the Clarabel solver.
    """
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC, ExpCone, PowCone3D]
    try:
        import clarabel
        if Version(clarabel.__version__) >= Version('0.5.0'):
            SUPPORTED_CONSTRAINTS.append(PSD)
    except (ModuleNotFoundError, TypeError):
        pass
    STATUS_MAP = {'Solved': s.OPTIMAL, 'PrimalInfeasible': s.INFEASIBLE, 'DualInfeasible': s.UNBOUNDED, 'AlmostSolved': s.OPTIMAL_INACCURATE, 'AlmostPrimalInfeasible': s.INFEASIBLE_INACCURATE, 'AlmostDualInfeasible': s.UNBOUNDED_INACCURATE, 'MaxIterations': s.USER_LIMIT, 'MaxTime': s.USER_LIMIT, 'NumericalError': s.SOLVER_ERROR, 'InsufficientProgress': s.SOLVER_ERROR}
    EXP_CONE_ORDER = [0, 1, 2]

    def name(self):
        """The name of the solver.
        """
        return 'CLARABEL'

    def import_solver(self) -> None:
        """Imports the solver.
        """
        import clarabel

    def supports_quad_obj(self) -> bool:
        """Clarabel supports quadratic objective with any combination 
        of conic constraints.
        """
        return True

    @staticmethod
    def psd_format_mat(constr):
        """Return a linear operator to multiply by PSD constraint coefficients.

        Special cases PSD constraints, as Clarabel expects constraints to be
        imposed on the upper triangular part of the variable matrix with
        symmetric scaling (i.e. off-diagonal sqrt(2) scalinig) applied.

        """
        rows = cols = constr.expr.shape[0]
        entries = rows * (cols + 1) // 2
        row_arr = np.arange(0, entries)
        upper_diag_indices = np.triu_indices(rows)
        col_arr = np.sort(np.ravel_multi_index(upper_diag_indices, (rows, cols), order='F'))
        val_arr = np.zeros((rows, cols))
        val_arr[upper_diag_indices] = np.sqrt(2)
        np.fill_diagonal(val_arr, 1.0)
        val_arr = np.ravel(val_arr, order='F')
        val_arr = val_arr[np.nonzero(val_arr)]
        shape = (entries, rows * cols)
        scaled_upper_tri = sp.csc_matrix((val_arr, (row_arr, col_arr)), shape)
        idx = np.arange(rows * cols)
        val_symm = 0.5 * np.ones(2 * rows * cols)
        K = idx.reshape((rows, cols))
        row_symm = np.append(idx, np.ravel(K, order='F'))
        col_symm = np.append(idx, np.ravel(K.T, order='F'))
        symm_matrix = sp.csc_matrix((val_symm, (row_symm, col_symm)))
        return scaled_upper_tri @ symm_matrix

    @staticmethod
    def extract_dual_value(result_vec, offset, constraint):
        """Extracts the dual value for constraint starting at offset.
        """
        if isinstance(constraint, PSD):
            dim = constraint.shape[0]
            upper_tri_dim = dim * (dim + 1) >> 1
            new_offset = offset + upper_tri_dim
            upper_tri = result_vec[offset:new_offset]
            full = triu_to_full(upper_tri, dim)
            return (full, new_offset)
        else:
            return utilities.extract_dual_value(result_vec, offset, constraint)

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        attr = {}
        status = self.STATUS_MAP[str(solution.status)]
        attr[s.SOLVE_TIME] = solution.solve_time
        attr[s.NUM_ITERS] = solution.iterations
        if status in s.SOLUTION_PRESENT:
            primal_val = solution.obj_val
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[CLARABEL.VAR_ID]: solution.x}
            eq_dual_vars = utilities.get_dual_values(solution.z[:inverse_data[ConicSolver.DIMS].zero], self.extract_dual_value, inverse_data[CLARABEL.EQ_CONSTR])
            ineq_dual_vars = utilities.get_dual_values(solution.z[inverse_data[ConicSolver.DIMS].zero:], self.extract_dual_value, inverse_data[CLARABEL.NEQ_CONSTR])
            dual_vars = {}
            dual_vars.update(eq_dual_vars)
            dual_vars.update(ineq_dual_vars)
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    @staticmethod
    def parse_solver_opts(verbose, opts):
        import clarabel
        settings = clarabel.DefaultSettings()
        settings.verbose = verbose
        if 'use_quad_obj' in opts:
            del opts['use_quad_obj']
        for opt in opts.keys():
            try:
                settings.__setattr__(opt, opts[opt])
            except TypeError as e:
                raise TypeError(f"Clarabel: Incorrect type for setting '{opt}'.") from e
            except AttributeError as e:
                raise TypeError(f"Clarabel: unrecognized solver setting '{opt}'.") from e
        return settings

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        warm_start : Bool
            Whether to warm_start Clarabel.
            PJG: From SCS.   We don't support this, not sure if relevant
        verbose : Bool
            Control the verbosity.
        solver_opts : dict
            Clarabel-specific solver options.

        Returns
        -------
        The result returned by a call to clarabel.solve().
        """
        import clarabel
        A = data[s.A]
        b = data[s.B]
        c = data[s.C]
        if s.P in data:
            P = data[s.P]
        else:
            nvars = c.size
            P = sp.csc_matrix((nvars, nvars))
        cones = dims_to_solver_cones(data[ConicSolver.DIMS])

        def solve(_solver_opts):
            _settings = CLARABEL.parse_solver_opts(verbose, _solver_opts)
            _solver = clarabel.DefaultSolver(P, c, A, b, cones, _settings)
            _results = _solver.solve()
            return (_results, _results.status)
        results, status = solve(solver_opts)
        if solver_cache is not None and self.STATUS_MAP[str(status)]:
            solver_cache[self.name()] = results
        return results