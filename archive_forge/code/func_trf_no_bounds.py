import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd, qr
from scipy.sparse.linalg import lsmr
from scipy.optimize import OptimizeResult
from .common import (
def trf_no_bounds(fun, jac, x0, f0, J0, ftol, xtol, gtol, max_nfev, x_scale, loss_function, tr_solver, tr_options, verbose):
    x = x0.copy()
    f = f0
    f_true = f.copy()
    nfev = 1
    J = J0
    njev = 1
    m, n = J.shape
    if loss_function is not None:
        rho = loss_function(f)
        cost = 0.5 * np.sum(rho[0])
        J, f = scale_for_robust_loss_function(J, f, rho)
    else:
        cost = 0.5 * np.dot(f, f)
    g = compute_grad(J, f)
    jac_scale = isinstance(x_scale, str) and x_scale == 'jac'
    if jac_scale:
        scale, scale_inv = compute_jac_scale(J)
    else:
        scale, scale_inv = (x_scale, 1 / x_scale)
    Delta = norm(x0 * scale_inv)
    if Delta == 0:
        Delta = 1.0
    if tr_solver == 'lsmr':
        reg_term = 0
        damp = tr_options.pop('damp', 0.0)
        regularize = tr_options.pop('regularize', True)
    if max_nfev is None:
        max_nfev = x0.size * 100
    alpha = 0.0
    termination_status = None
    iteration = 0
    step_norm = None
    actual_reduction = None
    if verbose == 2:
        print_header_nonlinear()
    while True:
        g_norm = norm(g, ord=np.inf)
        if g_norm < gtol:
            termination_status = 1
        if verbose == 2:
            print_iteration_nonlinear(iteration, nfev, cost, actual_reduction, step_norm, g_norm)
        if termination_status is not None or nfev == max_nfev:
            break
        d = scale
        g_h = d * g
        if tr_solver == 'exact':
            J_h = J * d
            U, s, V = svd(J_h, full_matrices=False)
            V = V.T
            uf = U.T.dot(f)
        elif tr_solver == 'lsmr':
            J_h = right_multiplied_operator(J, d)
            if regularize:
                a, b = build_quadratic_1d(J_h, g_h, -g_h)
                to_tr = Delta / norm(g_h)
                ag_value = minimize_quadratic_1d(a, b, 0, to_tr)[1]
                reg_term = -ag_value / Delta ** 2
            damp_full = (damp ** 2 + reg_term) ** 0.5
            gn_h = lsmr(J_h, f, damp=damp_full, **tr_options)[0]
            S = np.vstack((g_h, gn_h)).T
            S, _ = qr(S, mode='economic')
            JS = J_h.dot(S)
            B_S = np.dot(JS.T, JS)
            g_S = S.T.dot(g_h)
        actual_reduction = -1
        while actual_reduction <= 0 and nfev < max_nfev:
            if tr_solver == 'exact':
                step_h, alpha, n_iter = solve_lsq_trust_region(n, m, uf, s, V, Delta, initial_alpha=alpha)
            elif tr_solver == 'lsmr':
                p_S, _ = solve_trust_region_2d(B_S, g_S, Delta)
                step_h = S.dot(p_S)
            predicted_reduction = -evaluate_quadratic(J_h, g_h, step_h)
            step = d * step_h
            x_new = x + step
            f_new = fun(x_new)
            nfev += 1
            step_h_norm = norm(step_h)
            if not np.all(np.isfinite(f_new)):
                Delta = 0.25 * step_h_norm
                continue
            if loss_function is not None:
                cost_new = loss_function(f_new, cost_only=True)
            else:
                cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new
            Delta_new, ratio = update_tr_radius(Delta, actual_reduction, predicted_reduction, step_h_norm, step_h_norm > 0.95 * Delta)
            step_norm = norm(step)
            termination_status = check_termination(actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)
            if termination_status is not None:
                break
            alpha *= Delta / Delta_new
            Delta = Delta_new
        if actual_reduction > 0:
            x = x_new
            f = f_new
            f_true = f.copy()
            cost = cost_new
            J = jac(x, f)
            njev += 1
            if loss_function is not None:
                rho = loss_function(f)
                J, f = scale_for_robust_loss_function(J, f, rho)
            g = compute_grad(J, f)
            if jac_scale:
                scale, scale_inv = compute_jac_scale(J, scale_inv)
        else:
            step_norm = 0
            actual_reduction = 0
        iteration += 1
    if termination_status is None:
        termination_status = 0
    active_mask = np.zeros_like(x)
    return OptimizeResult(x=x, cost=cost, fun=f_true, jac=J, grad=g, optimality=g_norm, active_mask=active_mask, nfev=nfev, njev=njev, status=termination_status)