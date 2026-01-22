import math
import warnings
import numpy as np
import scipy.linalg
from ._optimize import (_check_unknown_options, _status_message,
from scipy.optimize._hessian_update_strategy import HessianUpdateStrategy
from scipy.optimize._differentiable_functions import FD_METHODS
def _minimize_trust_region(fun, x0, args=(), jac=None, hess=None, hessp=None, subproblem=None, initial_trust_radius=1.0, max_trust_radius=1000.0, eta=0.15, gtol=0.0001, maxiter=None, disp=False, return_all=False, callback=None, inexact=True, **unknown_options):
    """
    Minimization of scalar function of one or more variables using a
    trust-region algorithm.

    Options for the trust-region algorithm are:
        initial_trust_radius : float
            Initial trust radius.
        max_trust_radius : float
            Never propose steps that are longer than this value.
        eta : float
            Trust region related acceptance stringency for proposed steps.
        gtol : float
            Gradient norm must be less than `gtol`
            before successful termination.
        maxiter : int
            Maximum number of iterations to perform.
        disp : bool
            If True, print convergence message.
        inexact : bool
            Accuracy to solve subproblems. If True requires less nonlinear
            iterations, but more vector products. Only effective for method
            trust-krylov.

    This function is called by the `minimize` function.
    It is not supposed to be called directly.
    """
    _check_unknown_options(unknown_options)
    if jac is None:
        raise ValueError('Jacobian is currently required for trust-region methods')
    if hess is None and hessp is None:
        raise ValueError('Either the Hessian or the Hessian-vector product is currently required for trust-region methods')
    if subproblem is None:
        raise ValueError('A subproblem solving strategy is required for trust-region methods')
    if not 0 <= eta < 0.25:
        raise Exception('invalid acceptance stringency')
    if max_trust_radius <= 0:
        raise Exception('the max trust radius must be positive')
    if initial_trust_radius <= 0:
        raise ValueError('the initial trust radius must be positive')
    if initial_trust_radius >= max_trust_radius:
        raise ValueError('the initial trust radius must be less than the max trust radius')
    x0 = np.asarray(x0).flatten()
    sf = _prepare_scalar_function(fun, x0, jac=jac, hess=hess, args=args)
    fun = sf.fun
    jac = sf.grad
    if callable(hess):
        hess = sf.hess
    elif callable(hessp):
        pass
    elif hess in FD_METHODS or isinstance(hess, HessianUpdateStrategy):
        hess = None

        def hessp(x, p, *args):
            return sf.hess(x).dot(p)
    else:
        raise ValueError('Either the Hessian or the Hessian-vector product is currently required for trust-region methods')
    nhessp, hessp = _wrap_function(hessp, args)
    if maxiter is None:
        maxiter = len(x0) * 200
    warnflag = 0
    trust_radius = initial_trust_radius
    x = x0
    if return_all:
        allvecs = [x]
    m = subproblem(x, fun, jac, hess, hessp)
    k = 0
    while m.jac_mag >= gtol:
        try:
            p, hits_boundary = m.solve(trust_radius)
        except np.linalg.LinAlgError:
            warnflag = 3
            break
        predicted_value = m(p)
        x_proposed = x + p
        m_proposed = subproblem(x_proposed, fun, jac, hess, hessp)
        actual_reduction = m.fun - m_proposed.fun
        predicted_reduction = m.fun - predicted_value
        if predicted_reduction <= 0:
            warnflag = 2
            break
        rho = actual_reduction / predicted_reduction
        if rho < 0.25:
            trust_radius *= 0.25
        elif rho > 0.75 and hits_boundary:
            trust_radius = min(2 * trust_radius, max_trust_radius)
        if rho > eta:
            x = x_proposed
            m = m_proposed
        if return_all:
            allvecs.append(np.copy(x))
        k += 1
        intermediate_result = OptimizeResult(x=x, fun=m.fun)
        if _call_callback_maybe_halt(callback, intermediate_result):
            break
        if m.jac_mag < gtol:
            warnflag = 0
            break
        if k >= maxiter:
            warnflag = 1
            break
    status_messages = (_status_message['success'], _status_message['maxiter'], 'A bad approximation caused failure to predict improvement.', 'A linalg error occurred, such as a non-psd Hessian.')
    if disp:
        if warnflag == 0:
            print(status_messages[warnflag])
        else:
            warnings.warn(status_messages[warnflag], RuntimeWarning, stacklevel=3)
        print('         Current function value: %f' % m.fun)
        print('         Iterations: %d' % k)
        print('         Function evaluations: %d' % sf.nfev)
        print('         Gradient evaluations: %d' % sf.ngev)
        print('         Hessian evaluations: %d' % (sf.nhev + nhessp[0]))
    result = OptimizeResult(x=x, success=warnflag == 0, status=warnflag, fun=m.fun, jac=m.jac, nfev=sf.nfev, njev=sf.ngev, nhev=sf.nhev + nhessp[0], nit=k, message=status_messages[warnflag])
    if hess is not None:
        result['hess'] = m.hess
    if return_all:
        result['allvecs'] = allvecs
    return result