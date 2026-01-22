import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
def _scalar_optimization_prepare_result(work, res, res_work_pairs, active, shape, customize_result):
    res = res.copy()
    _scalar_optimization_update_active(work, res, res_work_pairs, active)
    shape = customize_result(res, shape)
    for key, val in res.items():
        res[key] = np.reshape(val, shape)[()]
    res['_order_keys'] = ['success'] + [i for i, j in res_work_pairs]
    return OptimizeResult(**res)