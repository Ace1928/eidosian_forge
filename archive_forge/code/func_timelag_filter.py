from decorator import decorator
import numpy as np
import scipy
import scipy.signal
import scipy.ndimage
import sklearn
import sklearn.cluster
import sklearn.feature_extraction
import sklearn.neighbors
from ._cache import cache
from . import util
from .filters import diagonal_filter
from .util.exceptions import ParameterError
from typing import Any, Callable, Optional, TypeVar, Union, overload
from typing_extensions import Literal
from ._typing import _WindowSpec, _FloatLike_co
def timelag_filter(function: _F, pad: bool=True, index: int=0) -> _F:
    """Apply a filter in the time-lag domain.

    This is primarily useful for adapting image filters to operate on
    `recurrence_to_lag` output.

    Using `timelag_filter` is equivalent to the following sequence of
    operations:

    >>> data_tl = librosa.segment.recurrence_to_lag(data)
    >>> data_filtered_tl = function(data_tl)
    >>> data_filtered = librosa.segment.lag_to_recurrence(data_filtered_tl)

    Parameters
    ----------
    function : callable
        The filtering function to wrap, e.g., `scipy.ndimage.median_filter`
    pad : bool
        Whether to zero-pad the structure feature matrix
    index : int >= 0
        If ``function`` accepts input data as a positional argument, it should be
        indexed by ``index``

    Returns
    -------
    wrapped_function : callable
        A new filter function which applies in time-lag space rather than
        time-time space.

    Examples
    --------
    Apply a 31-bin median filter to the diagonal of a recurrence matrix.
    With default, parameters, this corresponds to a time window of about
    0.72 seconds.

    >>> y, sr = librosa.load(librosa.ex('nutcracker'), duration=30)
    >>> chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    >>> chroma_stack = librosa.feature.stack_memory(chroma, n_steps=3, delay=3)
    >>> rec = librosa.segment.recurrence_matrix(chroma_stack)
    >>> from scipy.ndimage import median_filter
    >>> diagonal_median = librosa.segment.timelag_filter(median_filter)
    >>> rec_filtered = diagonal_median(rec, size=(1, 31), mode='mirror')

    Or with affinity weights

    >>> rec_aff = librosa.segment.recurrence_matrix(chroma_stack, mode='affinity')
    >>> rec_aff_fil = diagonal_median(rec_aff, size=(1, 31), mode='mirror')

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    >>> librosa.display.specshow(rec, y_axis='s', x_axis='s', ax=ax[0, 0])
    >>> ax[0, 0].set(title='Raw recurrence matrix')
    >>> ax[0, 0].label_outer()
    >>> librosa.display.specshow(rec_filtered, y_axis='s', x_axis='s', ax=ax[0, 1])
    >>> ax[0, 1].set(title='Filtered recurrence matrix')
    >>> ax[0, 1].label_outer()
    >>> librosa.display.specshow(rec_aff, x_axis='s', y_axis='s',
    ...                          cmap='magma_r', ax=ax[1, 0])
    >>> ax[1, 0].set(title='Raw affinity matrix')
    >>> librosa.display.specshow(rec_aff_fil, x_axis='s', y_axis='s',
    ...                          cmap='magma_r', ax=ax[1, 1])
    >>> ax[1, 1].set(title='Filtered affinity matrix')
    >>> ax[1, 1].label_outer()
    """

    def __my_filter(wrapped_f, *args, **kwargs):
        """Wrap the filter with lag conversions"""
        args = list(args)
        args[index] = recurrence_to_lag(args[index], pad=pad)
        result = wrapped_f(*args, **kwargs)
        return lag_to_recurrence(result)
    return decorator(__my_filter, function)