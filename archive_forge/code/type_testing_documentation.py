import numpy
import cupy
Return True if x is a not complex type or an array of complex numbers.

    The type of the input is checked, not the value. So even if the input
    has an imaginary part equal to zero, `isrealobj` evaluates to False
    if the data type is complex.

    Args:
        x (cupy.ndarray): The input can be of any type and shape.

    Returns:
        bool: The return value, False if ``x`` is of a complex type.

    .. seealso::
        :func:`iscomplexobj`, :func:`isreal`

    Examples
    --------
    >>> cupy.isrealobj(cupy.array([3, 1+0j, True]))
    False
    >>> cupy.isrealobj(cupy.array([3, 1, True]))
    True

    