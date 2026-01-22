from functools import partial
import numpy as np
from . import _catboost
def _current_params(metric_obj, override_only):
    params_with_defaults = metric_obj.params_with_defaults()
    param_info = {}
    for param in sorted(metric_obj._params):
        value = getattr(metric_obj, param)
        if param == 'hints' and value == '':
            continue
        if override_only:
            default_value = params_with_defaults[param]['default_value']
            if default_value is None and value is None or (default_value is not None and default_value == value):
                continue
        param_info[param] = value
    return param_info