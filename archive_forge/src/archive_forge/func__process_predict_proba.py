import numpy as np
from ..base import is_classifier
from .multiclass import type_of_target
from .validation import _check_response_method, check_is_fitted
def _process_predict_proba(*, y_pred, target_type, classes, pos_label):
    """Get the response values when the response method is `predict_proba`.

    This function process the `y_pred` array in the binary and multi-label cases.
    In the binary case, it selects the column corresponding to the positive
    class. In the multi-label case, it stacks the predictions if they are not
    in the "compressed" format `(n_samples, n_outputs)`.

    Parameters
    ----------
    y_pred : ndarray
        Output of `estimator.predict_proba`. The shape depends on the target type:

        - for binary classification, it is a 2d array of shape `(n_samples, 2)`;
        - for multiclass classification, it is a 2d array of shape
          `(n_samples, n_classes)`;
        - for multilabel classification, it is either a list of 2d arrays of shape
          `(n_samples, 2)` (e.g. `RandomForestClassifier` or `KNeighborsClassifier`) or
          an array of shape `(n_samples, n_outputs)` (e.g. `MLPClassifier` or
          `RidgeClassifier`).

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
    if target_type == 'binary' and y_pred.shape[1] < 2:
        raise ValueError(f'Got predict_proba of shape {y_pred.shape}, but need classifier with two classes.')
    if target_type == 'binary':
        col_idx = np.flatnonzero(classes == pos_label)[0]
        return y_pred[:, col_idx]
    elif target_type == 'multilabel-indicator':
        if isinstance(y_pred, list):
            return np.vstack([p[:, -1] for p in y_pred]).T
        else:
            return y_pred
    return y_pred