import numpy
import numpy as np
from . import _ni_support
from . import _ni_label
from . import _nd_image
from . import _morphology
def value_indices(arr, *, ignore_value=None):
    """
    Find indices of each distinct value in given array.

    Parameters
    ----------
    arr : ndarray of ints
        Array containing integer values.
    ignore_value : int, optional
        This value will be ignored in searching the `arr` array. If not
        given, all values found will be included in output. Default
        is None.

    Returns
    -------
    indices : dictionary
        A Python dictionary of array indices for each distinct value. The
        dictionary is keyed by the distinct values, the entries are array
        index tuples covering all occurrences of the value within the
        array.

        This dictionary can occupy significant memory, usually several times
        the size of the input array.

    See Also
    --------
    label, maximum, median, minimum_position, extrema, sum, mean, variance,
    standard_deviation, numpy.where, numpy.unique

    Notes
    -----
    For a small array with few distinct values, one might use
    `numpy.unique()` to find all possible values, and ``(arr == val)`` to
    locate each value within that array. However, for large arrays,
    with many distinct values, this can become extremely inefficient,
    as locating each value would require a new search through the entire
    array. Using this function, there is essentially one search, with
    the indices saved for all distinct values.

    This is useful when matching a categorical image (e.g. a segmentation
    or classification) to an associated image of other data, allowing
    any per-class statistic(s) to then be calculated. Provides a
    more flexible alternative to functions like ``scipy.ndimage.mean()``
    and ``scipy.ndimage.variance()``.

    Some other closely related functionality, with different strengths and
    weaknesses, can also be found in ``scipy.stats.binned_statistic()`` and
    the `scikit-image <https://scikit-image.org/>`_ function
    ``skimage.measure.regionprops()``.

    Note for IDL users: this provides functionality equivalent to IDL's
    REVERSE_INDICES option (as per the IDL documentation for the
    `HISTOGRAM <https://www.l3harrisgeospatial.com/docs/histogram.html>`_
    function).

    .. versionadded:: 1.10.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import ndimage
    >>> a = np.zeros((6, 6), dtype=int)
    >>> a[2:4, 2:4] = 1
    >>> a[4, 4] = 1
    >>> a[:2, :3] = 2
    >>> a[0, 5] = 3
    >>> a
    array([[2, 2, 2, 0, 0, 3],
           [2, 2, 2, 0, 0, 0],
           [0, 0, 1, 1, 0, 0],
           [0, 0, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0]])
    >>> val_indices = ndimage.value_indices(a)

    The dictionary `val_indices` will have an entry for each distinct
    value in the input array.

    >>> val_indices.keys()
    dict_keys([0, 1, 2, 3])

    The entry for each value is an index tuple, locating the elements
    with that value.

    >>> ndx1 = val_indices[1]
    >>> ndx1
    (array([2, 2, 3, 3, 4]), array([2, 3, 2, 3, 4]))

    This can be used to index into the original array, or any other
    array with the same shape.

    >>> a[ndx1]
    array([1, 1, 1, 1, 1])

    If the zeros were to be ignored, then the resulting dictionary
    would no longer have an entry for zero.

    >>> val_indices = ndimage.value_indices(a, ignore_value=0)
    >>> val_indices.keys()
    dict_keys([1, 2, 3])

    """
    ignore_value_arr = numpy.zeros((1,), dtype=arr.dtype)
    ignoreIsNone = ignore_value is None
    if not ignoreIsNone:
        ignore_value_arr[0] = ignore_value_arr.dtype.type(ignore_value)
    val_indices = _nd_image.value_indices(arr, ignoreIsNone, ignore_value_arr)
    return val_indices