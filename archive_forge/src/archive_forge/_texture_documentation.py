import cupy
from cupy import _core
from cupy.cuda import texture
from cupy.cuda import runtime

    Apply an affine transformation.

    The method uses texture memory and supports only 2D and 3D float32 arrays
    without channel dimension.

    Args:
        data (cupy.ndarray): The input array or texture object.
        transformation_matrix (cupy.ndarray): Affine transformation matrix.
            Must be a homogeneous and have shape ``(ndim + 1, ndim + 1)``.
        output_shape (tuple of ints): Shape of output. If not specified,
            the input array shape is used. Default is None.
        output (cupy.ndarray or ~cupy.dtype): The array in which to place the
            output, or the dtype of the returned array. If not specified,
            creates the output array with shape of ``output_shape``. Default is
            None.
        interpolation (str): Specifies interpolation mode: ``'linear'`` or
            ``'nearest'``. Default is ``'linear'``.
        mode (str): Specifies addressing mode for points outside of the array:
            (`'constant'``, ``'nearest'``). Default is ``'constant'``.
        border_value: Specifies value to be used for coordinates outside
            of the array for ``'constant'`` mode. Default is 0.

    Returns:
        cupy.ndarray:
            The transformed input.

    .. seealso:: :func:`cupyx.scipy.ndimage.affine_transform`
    