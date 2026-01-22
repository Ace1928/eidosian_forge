import warnings
from collections import namedtuple
import numpy as np
from scipy import optimize, stats
from scipy._lib._util import check_random_state
def _get_fit_fun(dist, data, guessed_params, fixed_params):
    shape_names = [] if dist.shapes is None else dist.shapes.split(', ')
    param_names = shape_names + ['loc', 'scale']
    fparam_names = ['f' + name for name in param_names]
    all_fixed = not set(fparam_names).difference(fixed_params)
    guessed_shapes = [guessed_params.pop(x, None) for x in shape_names if x in guessed_params]
    if all_fixed:

        def fit_fun(data):
            return [fixed_params[name] for name in fparam_names]
    elif dist in _fit_funs:

        def fit_fun(data):
            params = _fit_funs[dist](data, **fixed_params)
            params = np.asarray(np.broadcast_arrays(*params))
            if params.ndim > 1:
                params = params[..., np.newaxis]
            return params
    else:

        def fit_fun_1d(data):
            return dist.fit(data, *guessed_shapes, **guessed_params, **fixed_params)

        def fit_fun(data):
            params = np.apply_along_axis(fit_fun_1d, axis=-1, arr=data)
            if params.ndim > 1:
                params = params.T[..., np.newaxis]
            return params
    return fit_fun