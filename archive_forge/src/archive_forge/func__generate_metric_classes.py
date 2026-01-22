from functools import partial
import numpy as np
from . import _catboost
def _generate_metric_classes():
    for metric_name, metric_param_sets in _catboost.AllMetricsParams().items():
        for param_set in metric_param_sets:
            derived_name = metric_name + param_set['_name_suffix']
            del param_set['_name_suffix']
            valid_params = {param: param_value['default_value'] if not param_value['is_mandatory'] else None for param, param_value in param_set.items()}
            is_mandatory_param = {param: param_value['is_mandatory'] for param, param_value in param_set.items()}
            if 'hints' not in valid_params:
                valid_params.update({'hints': ''})
                is_mandatory_param.update({'hints': False})
            globals()[derived_name] = _MetricGenerator(str(derived_name), (BuiltinMetric,), {'_valid_params': valid_params, '_is_mandatory_param': is_mandatory_param, '_underlying_metric_name': metric_name})
            globals()['__all__'].append(derived_name)