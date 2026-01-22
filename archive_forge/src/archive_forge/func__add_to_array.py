from warnings import warn
import numpy as np
from ._optimize import (_minimize_neldermead, _minimize_powell, _minimize_cg,
from ._trustregion_dogleg import _minimize_dogleg
from ._trustregion_ncg import _minimize_trust_ncg
from ._trustregion_krylov import _minimize_trust_krylov
from ._trustregion_exact import _minimize_trustregion_exact
from ._trustregion_constr import _minimize_trustregion_constr
from ._lbfgsb_py import _minimize_lbfgsb
from ._tnc import _minimize_tnc
from ._cobyla_py import _minimize_cobyla
from ._slsqp_py import _minimize_slsqp
from ._constraints import (old_bound_to_new, new_bounds_to_old,
from ._differentiable_functions import FD_METHODS
def _add_to_array(x_in, i_fixed, x_fixed):
    """Adds fixed variables back to an array"""
    i_free = ~i_fixed
    if x_in.ndim == 2:
        i_free = i_free[:, None] @ i_free[None, :]
    x_out = np.zeros_like(i_free, dtype=x_in.dtype)
    x_out[~i_free] = x_fixed
    x_out[i_free] = x_in.ravel()
    return x_out