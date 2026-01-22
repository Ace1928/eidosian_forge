from functools import partial
import numpy as np
from . import _catboost
def _del_param(metric_obj, name):
    """Validate a new parameter value in a created metric object."""
    if name not in metric_obj._valid_params:
        raise ValueError("Metric {} doesn't have a parameter {}.".format(metric_obj.__name__, name))
    if metric_obj._is_mandatory_param[name]:
        raise ValueError('Parameter {} is mandatory, cannot reset.'.format(name))
    value = metric_obj._valid_params[name]
    setattr(metric_obj, '_' + name, value)