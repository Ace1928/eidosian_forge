import cupy
from cupy import _core
Unpacks elements of a uint8 array into a binary-valued output array.

    This function currently does not support ``axis`` option.

    Args:
        a (cupy.ndarray): Input array.
        bitorder (str, optional): bit order to use when unpacking the array,
            allowed values are `'little'` and `'big'`. Defaults to `'big'`.

    Returns:
        cupy.ndarray: The unpacked array.

    .. seealso:: :func:`numpy.unpackbits`
    