import scipy.sparse as sps
import numpy as np
from .equality_constrained_sqp import equality_constrained_sqp
from scipy.sparse.linalg import LinearOperator
class BarrierSubproblem:
    """
    Barrier optimization problem:
        minimize fun(x) - barrier_parameter*sum(log(s))
        subject to: constr_eq(x)     = 0
                  constr_ineq(x) + s = 0
    """

    def __init__(self, x0, s0, fun, grad, lagr_hess, n_vars, n_ineq, n_eq, constr, jac, barrier_parameter, tolerance, enforce_feasibility, global_stop_criteria, xtol, fun0, grad0, constr_ineq0, jac_ineq0, constr_eq0, jac_eq0):
        self.n_vars = n_vars
        self.x0 = x0
        self.s0 = s0
        self.fun = fun
        self.grad = grad
        self.lagr_hess = lagr_hess
        self.constr = constr
        self.jac = jac
        self.barrier_parameter = barrier_parameter
        self.tolerance = tolerance
        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.enforce_feasibility = enforce_feasibility
        self.global_stop_criteria = global_stop_criteria
        self.xtol = xtol
        self.fun0 = self._compute_function(fun0, constr_ineq0, s0)
        self.grad0 = self._compute_gradient(grad0)
        self.constr0 = self._compute_constr(constr_ineq0, constr_eq0, s0)
        self.jac0 = self._compute_jacobian(jac_eq0, jac_ineq0, s0)
        self.terminate = False

    def update(self, barrier_parameter, tolerance):
        self.barrier_parameter = barrier_parameter
        self.tolerance = tolerance

    def get_slack(self, z):
        return z[self.n_vars:self.n_vars + self.n_ineq]

    def get_variables(self, z):
        return z[:self.n_vars]

    def function_and_constraints(self, z):
        """Returns barrier function and constraints at given point.

        For z = [x, s], returns barrier function:
            function(z) = fun(x) - barrier_parameter*sum(log(s))
        and barrier constraints:
            constraints(z) = [   constr_eq(x)     ]
                             [ constr_ineq(x) + s ]

        """
        x = self.get_variables(z)
        s = self.get_slack(z)
        f = self.fun(x)
        c_eq, c_ineq = self.constr(x)
        return (self._compute_function(f, c_ineq, s), self._compute_constr(c_ineq, c_eq, s))

    def _compute_function(self, f, c_ineq, s):
        s[self.enforce_feasibility] = -c_ineq[self.enforce_feasibility]
        log_s = [np.log(s_i) if s_i > 0 else -np.inf for s_i in s]
        return f - self.barrier_parameter * np.sum(log_s)

    def _compute_constr(self, c_ineq, c_eq, s):
        return np.hstack((c_eq, c_ineq + s))

    def scaling(self, z):
        """Returns scaling vector.
        Given by:
            scaling = [ones(n_vars), s]
        """
        s = self.get_slack(z)
        diag_elements = np.hstack((np.ones(self.n_vars), s))

        def matvec(vec):
            return diag_elements * vec
        return LinearOperator((self.n_vars + self.n_ineq, self.n_vars + self.n_ineq), matvec)

    def gradient_and_jacobian(self, z):
        """Returns scaled gradient.

        Return scaled gradient:
            gradient = [             grad(x)             ]
                       [ -barrier_parameter*ones(n_ineq) ]
        and scaled Jacobian matrix:
            jacobian = [  jac_eq(x)  0  ]
                       [ jac_ineq(x) S  ]
        Both of them scaled by the previously defined scaling factor.
        """
        x = self.get_variables(z)
        s = self.get_slack(z)
        g = self.grad(x)
        J_eq, J_ineq = self.jac(x)
        return (self._compute_gradient(g), self._compute_jacobian(J_eq, J_ineq, s))

    def _compute_gradient(self, g):
        return np.hstack((g, -self.barrier_parameter * np.ones(self.n_ineq)))

    def _compute_jacobian(self, J_eq, J_ineq, s):
        if self.n_ineq == 0:
            return J_eq
        elif sps.issparse(J_eq) or sps.issparse(J_ineq):
            J_eq = sps.csr_matrix(J_eq)
            J_ineq = sps.csr_matrix(J_ineq)
            return self._assemble_sparse_jacobian(J_eq, J_ineq, s)
        else:
            S = np.diag(s)
            zeros = np.zeros((self.n_eq, self.n_ineq))
            if sps.issparse(J_ineq):
                J_ineq = J_ineq.toarray()
            if sps.issparse(J_eq):
                J_eq = J_eq.toarray()
            return np.block([[J_eq, zeros], [J_ineq, S]])

    def _assemble_sparse_jacobian(self, J_eq, J_ineq, s):
        """Assemble sparse Jacobian given its components.

        Given ``J_eq``, ``J_ineq`` and ``s`` returns:
            jacobian = [ J_eq,     0     ]
                       [ J_ineq, diag(s) ]

        It is equivalent to:
            sps.bmat([[ J_eq,   None    ],
                      [ J_ineq, diag(s) ]], "csr")
        but significantly more efficient for this
        given structure.
        """
        n_vars, n_ineq, n_eq = (self.n_vars, self.n_ineq, self.n_eq)
        J_aux = sps.vstack([J_eq, J_ineq], 'csr')
        indptr, indices, data = (J_aux.indptr, J_aux.indices, J_aux.data)
        new_indptr = indptr + np.hstack((np.zeros(n_eq, dtype=int), np.arange(n_ineq + 1, dtype=int)))
        size = indices.size + n_ineq
        new_indices = np.empty(size)
        new_data = np.empty(size)
        mask = np.full(size, False, bool)
        mask[new_indptr[-n_ineq:] - 1] = True
        new_indices[mask] = n_vars + np.arange(n_ineq)
        new_indices[~mask] = indices
        new_data[mask] = s
        new_data[~mask] = data
        J = sps.csr_matrix((new_data, new_indices, new_indptr), (n_eq + n_ineq, n_vars + n_ineq))
        return J

    def lagrangian_hessian_x(self, z, v):
        """Returns Lagrangian Hessian (in relation to `x`) -> Hx"""
        x = self.get_variables(z)
        v_eq = v[:self.n_eq]
        v_ineq = v[self.n_eq:self.n_eq + self.n_ineq]
        lagr_hess = self.lagr_hess
        return lagr_hess(x, v_eq, v_ineq)

    def lagrangian_hessian_s(self, z, v):
        """Returns scaled Lagrangian Hessian (in relation to`s`) -> S Hs S"""
        s = self.get_slack(z)
        primal = self.barrier_parameter
        primal_dual = v[-self.n_ineq:] * s
        return np.where(v[-self.n_ineq:] > 0, primal_dual, primal)

    def lagrangian_hessian(self, z, v):
        """Returns scaled Lagrangian Hessian"""
        Hx = self.lagrangian_hessian_x(z, v)
        if self.n_ineq > 0:
            S_Hs_S = self.lagrangian_hessian_s(z, v)

        def matvec(vec):
            vec_x = self.get_variables(vec)
            vec_s = self.get_slack(vec)
            if self.n_ineq > 0:
                return np.hstack((Hx.dot(vec_x), S_Hs_S * vec_s))
            else:
                return Hx.dot(vec_x)
        return LinearOperator((self.n_vars + self.n_ineq, self.n_vars + self.n_ineq), matvec)

    def stop_criteria(self, state, z, last_iteration_failed, optimality, constr_violation, trust_radius, penalty, cg_info):
        """Stop criteria to the barrier problem.
        The criteria here proposed is similar to formula (2.3)
        from [1]_, p.879.
        """
        x = self.get_variables(z)
        if self.global_stop_criteria(state, x, last_iteration_failed, trust_radius, penalty, cg_info, self.barrier_parameter, self.tolerance):
            self.terminate = True
            return True
        else:
            g_cond = optimality < self.tolerance and constr_violation < self.tolerance
            x_cond = trust_radius < self.xtol
            return g_cond or x_cond