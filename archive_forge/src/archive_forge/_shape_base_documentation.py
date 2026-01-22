from numpy.lib import index_tricks
import cupy
from cupy._core import internal
Apply a function to 1-D slices along the given axis.

    Args:
        func1d (function (M,) -> (Nj...)): This function should accept 1-D
            arrays. It is applied to 1-D slices of ``arr`` along the specified
            axis. It must return a 1-D ``cupy.ndarray``.
        axis (integer): Axis along which ``arr`` is sliced.
        arr (cupy.ndarray (Ni..., M, Nk...)): Input array.
        args: Additional arguments for ``func1d``.
        kwargs: Additional keyword arguments for ``func1d``.

    Returns:
        cupy.ndarray: The output array. The shape of ``out`` is identical to
        the shape of ``arr``, except along the ``axis`` dimension. This
        axis is removed, and replaced with new dimensions equal to the
        shape of the return value of ``func1d``. So if ``func1d`` returns a
        scalar ``out`` will have one fewer dimensions than ``arr``.

    .. seealso:: :func:`numpy.apply_along_axis`
    