from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _call_metric_fn(metric_fn, features, labels, predictions, config):
    """Calls metric fn with proper arguments."""
    metric_fn_args = function_utils.fn_args(metric_fn)
    kwargs = {}
    if 'features' in metric_fn_args:
        kwargs['features'] = features
    if 'labels' in metric_fn_args:
        kwargs['labels'] = labels
    if 'predictions' in metric_fn_args:
        kwargs['predictions'] = predictions
    if 'config' in metric_fn_args:
        kwargs['config'] = config
    return metric_fn(**kwargs)