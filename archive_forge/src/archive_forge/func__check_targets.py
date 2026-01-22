import warnings
from numbers import Integral, Real
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.special import xlogy
from ..exceptions import UndefinedMetricWarning
from ..preprocessing import LabelBinarizer, LabelEncoder
from ..utils import (
from ..utils._array_api import _union1d, _weighted_sum, get_namespace
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
from ..utils.extmath import _nanaverage
from ..utils.multiclass import type_of_target, unique_labels
from ..utils.sparsefuncs import count_nonzero
from ..utils.validation import _check_pos_label_consistency, _num_samples
def _check_targets(y_true, y_pred):
    """Check that y_true and y_pred belong to the same classification task.

    This converts multiclass or binary types to a common shape, and raises a
    ValueError for a mix of multilabel and multiclass targets, a mix of
    multilabel formats, for the presence of continuous-valued or multioutput
    targets, or for targets of different lengths.

    Column vectors are squeezed to 1d, while multilabel formats are returned
    as CSR sparse label indicators.

    Parameters
    ----------
    y_true : array-like

    y_pred : array-like

    Returns
    -------
    type_true : one of {'multilabel-indicator', 'multiclass', 'binary'}
        The type of the true target data, as output by
        ``utils.multiclass.type_of_target``.

    y_true : array or indicator matrix

    y_pred : array or indicator matrix
    """
    check_consistent_length(y_true, y_pred)
    type_true = type_of_target(y_true, input_name='y_true')
    type_pred = type_of_target(y_pred, input_name='y_pred')
    y_type = {type_true, type_pred}
    if y_type == {'binary', 'multiclass'}:
        y_type = {'multiclass'}
    if len(y_type) > 1:
        raise ValueError("Classification metrics can't handle a mix of {0} and {1} targets".format(type_true, type_pred))
    y_type = y_type.pop()
    if y_type not in ['binary', 'multiclass', 'multilabel-indicator']:
        raise ValueError('{0} is not supported'.format(y_type))
    if y_type in ['binary', 'multiclass']:
        xp, _ = get_namespace(y_true, y_pred)
        y_true = column_or_1d(y_true)
        y_pred = column_or_1d(y_pred)
        if y_type == 'binary':
            try:
                unique_values = _union1d(y_true, y_pred, xp)
            except TypeError as e:
                raise TypeError(f'Labels in y_true and y_pred should be of the same type. Got y_true={xp.unique(y_true)} and y_pred={xp.unique(y_pred)}. Make sure that the predictions provided by the classifier coincides with the true labels.') from e
            if unique_values.shape[0] > 2:
                y_type = 'multiclass'
    if y_type.startswith('multilabel'):
        y_true = csr_matrix(y_true)
        y_pred = csr_matrix(y_pred)
        y_type = 'multilabel-indicator'
    return (y_type, y_true, y_pred)