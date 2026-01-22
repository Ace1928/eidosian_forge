import warnings
from collections.abc import Sequence
from itertools import chain
import numpy as np
from scipy.sparse import issparse
from ..utils._array_api import get_namespace
from ..utils.fixes import VisibleDeprecationWarning
from .validation import _assert_all_finite, check_array
def _check_partial_fit_first_call(clf, classes=None):
    """Private helper function for factorizing common classes param logic.

    Estimators that implement the ``partial_fit`` API need to be provided with
    the list of possible classes at the first call to partial_fit.

    Subsequent calls to partial_fit should check that ``classes`` is still
    consistent with a previous value of ``clf.classes_`` when provided.

    This function returns True if it detects that this was the first call to
    ``partial_fit`` on ``clf``. In that case the ``classes_`` attribute is also
    set on ``clf``.

    """
    if getattr(clf, 'classes_', None) is None and classes is None:
        raise ValueError('classes must be passed on the first call to partial_fit.')
    elif classes is not None:
        if getattr(clf, 'classes_', None) is not None:
            if not np.array_equal(clf.classes_, unique_labels(classes)):
                raise ValueError('`classes=%r` is not the same as on last call to partial_fit, was: %r' % (classes, clf.classes_))
        else:
            clf.classes_ = unique_labels(classes)
            return True
    return False