from __future__ import annotations
import re
import numpy as np
from tlz import concat, merge, unique
from dask.array.core import Array, apply_infer_dtype, asarray, blockwise, getitem
from dask.array.utils import meta_from_array
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph
class gufunc:
    """
    Binds `pyfunc` into ``dask.array.apply_gufunc`` when called.

    Parameters
    ----------
    pyfunc : callable
        Function to call like ``func(*args, **kwargs)`` on input arrays
        (``*args``) that returns an array or tuple of arrays. If multiple
        arguments with non-matching dimensions are supplied, this function is
        expected to vectorize (broadcast) over axes of positional arguments in
        the style of NumPy universal functions [1]_ (if this is not the case,
        set ``vectorize=True``). If this function returns multiple outputs,
        ``output_core_dims`` has to be set as well.
    signature : String, keyword only
        Specifies what core dimensions are consumed and produced by ``func``.
        According to the specification of numpy.gufunc signature [2]_
    axes: List of tuples, optional, keyword only
        A list of tuples with indices of axes a generalized ufunc should operate on.
        For instance, for a signature of ``"(i,j),(j,k)->(i,k)"`` appropriate for
        matrix multiplication, the base elements are two-dimensional matrices
        and these are taken to be stored in the two last axes of each argument. The
        corresponding axes keyword would be ``[(-2, -1), (-2, -1), (-2, -1)]``.
        For simplicity, for generalized ufuncs that operate on 1-dimensional arrays
        (vectors), a single integer is accepted instead of a single-element tuple,
        and for generalized ufuncs for which all outputs are scalars, the output
        tuples can be omitted.
    axis: int, optional, keyword only
        A single axis over which a generalized ufunc should operate. This is a short-cut
        for ufuncs that operate over a single, shared core dimension, equivalent to passing
        in axes with entries of (axis,) for each single-core-dimension argument and ``()`` for
        all others. For instance, for a signature ``"(i),(i)->()"``, it is equivalent to passing
        in ``axes=[(axis,), (axis,), ()]``.
    keepdims: bool, optional, keyword only
        If this is set to True, axes which are reduced over will be left in the result as
        a dimension with size one, so that the result will broadcast correctly against the
        inputs. This option can only be used for generalized ufuncs that operate on inputs
        that all have the same number of core dimensions and with outputs that have no core
        dimensions , i.e., with signatures like ``"(i),(i)->()"`` or ``"(m,m)->()"``.
        If used, the location of the dimensions in the output can be controlled with axes
        and axis.
    output_dtypes : Optional, dtype or list of dtypes, keyword only
        Valid numpy dtype specification or list thereof.
        If not given, a call of ``func`` with a small set of data
        is performed in order to try to automatically determine the
        output dtypes.
    output_sizes : dict, optional, keyword only
        Optional mapping from dimension names to sizes for outputs. Only used if
        new core dimensions (not found on inputs) appear on outputs.
    vectorize: bool, keyword only
        If set to ``True``, ``np.vectorize`` is applied to ``func`` for
        convenience. Defaults to ``False``.
    allow_rechunk: Optional, bool, keyword only
        Allows rechunking, otherwise chunk sizes need to match and core
        dimensions are to consist only of one chunk.
        Warning: enabling this can increase memory usage significantly.
        Defaults to ``False``.
    meta: Optional, tuple, keyword only
        tuple of empty ndarrays describing the shape and dtype of the output of the gufunc.
        Defaults to ``None``.

    Returns
    -------
    Wrapped function

    Examples
    --------
    >>> import dask.array as da
    >>> import numpy as np
    >>> a = da.random.normal(size=(10,20,30), chunks=(5, 10, 30))
    >>> def stats(x):
    ...     return np.mean(x, axis=-1), np.std(x, axis=-1)
    >>> gustats = da.gufunc(stats, signature="(i)->(),()", output_dtypes=(float, float))
    >>> mean, std = gustats(a)
    >>> mean.compute().shape
    (10, 20)

    >>> a = da.random.normal(size=(   20,30), chunks=(10, 30))
    >>> b = da.random.normal(size=(10, 1,40), chunks=(5, 1, 40))
    >>> def outer_product(x, y):
    ...     return np.einsum("i,j->ij", x, y)
    >>> guouter_product = da.gufunc(outer_product, signature="(i),(j)->(i,j)", output_dtypes=float, vectorize=True)
    >>> c = guouter_product(a, b)
    >>> c.compute().shape
    (10, 20, 30, 40)

    >>> a = da.ones((1, 5, 10), chunks=(-1, -1, -1))
    >>> def stats(x):
    ...     return np.atleast_1d(x.mean()), np.atleast_1d(x.max())
    >>> meta = (np.array((), dtype=np.float64), np.array((), dtype=np.float64))
    >>> gustats = da.gufunc(stats, signature="(i,j)->(),()", meta=meta)
    >>> result = gustats(a)
    >>> result[0].compute().shape
    (1,)
    >>> result[1].compute().shape
    (1,)

    References
    ----------
    .. [1] https://docs.scipy.org/doc/numpy/reference/ufuncs.html
    .. [2] https://docs.scipy.org/doc/numpy/reference/c-api/generalized-ufuncs.html
    """

    def __init__(self, pyfunc, *, signature=None, vectorize=False, axes=None, axis=None, keepdims=False, output_sizes=None, output_dtypes=None, allow_rechunk=False, meta=None):
        self.pyfunc = pyfunc
        self.signature = signature
        self.vectorize = vectorize
        self.axes = axes
        self.axis = axis
        self.keepdims = keepdims
        self.output_sizes = output_sizes
        self.output_dtypes = output_dtypes
        self.allow_rechunk = allow_rechunk
        self.meta = meta
        self.__doc__ = "\n        Bound ``dask.array.gufunc``\n        func: ``{func}``\n        signature: ``'{signature}'``\n\n        Parameters\n        ----------\n        *args : numpy/dask arrays or scalars\n            Arrays to which to apply to ``func``. Core dimensions as specified in\n            ``signature`` need to come last.\n        **kwargs : dict\n            Extra keyword arguments to pass to ``func``\n\n        Returns\n        -------\n        Single dask.array.Array or tuple of dask.array.Array\n        ".format(func=str(self.pyfunc), signature=self.signature)

    def __call__(self, *args, allow_rechunk=False, **kwargs):
        return apply_gufunc(self.pyfunc, self.signature, *args, vectorize=self.vectorize, axes=self.axes, axis=self.axis, keepdims=self.keepdims, output_sizes=self.output_sizes, output_dtypes=self.output_dtypes, allow_rechunk=self.allow_rechunk or allow_rechunk, meta=self.meta, **kwargs)