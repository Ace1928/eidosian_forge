import numpy as np
from ..base import is_classifier
from .multiclass import type_of_target
from .validation import _check_response_method, check_is_fitted
def _process_decision_function(*, y_pred, target_type, classes, pos_label):
    """Get the response values when the response method is `decision_function`.

    This function process the `y_pred` array in the binary and multi-label cases.
    In the binary case, it inverts the sign of the score if the positive label
    is not `classes[1]`. In the multi-label case, it stacks the predictions if
    they are not in the "compressed" format `(n_samples, n_outputs)`.

    Parameters
    ----------
    y_pred : ndarray
        Output of `estimator.predict_proba`. The shape depends on the target type:

        - for binary classification, it is a 1d array of shape `(n_samples,)` where the
          sign is assuming that `classes[1]` is the positive class;
        - for multiclass classification, it is a 2d array of shape
          `(n_samples, n_classes)`;
        - for multilabel classification, it is a 2d array of shape `(n_samples,
          n_outputs)`.

    target_type : {"binary", "multiclass", "multilabel-indicator"}
        Type of the target.

    classes : ndarray of shape (n_classes,) or list of such arrays
        Class labels as reported by `estimator.classes_`.

    pos_label : int, float, bool or str
        Only used with binary and multiclass targets.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,), (n_samples, n_classes) or             (n_samples, n_output)
        Compressed predictions format as requested by the metrics.
    """
    if target_type == 'binary' and pos_label == classes[0]:
        return -1 * y_pred
    return y_pred