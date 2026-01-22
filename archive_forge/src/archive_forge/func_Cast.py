from ._internal import NDArrayBase
from ..base import _Null
def Cast(data=None, dtype=_Null, out=None, name=None, **kwargs):
    """Casts all elements of the input to a new type.

    .. note:: ``Cast`` is deprecated. Use ``cast`` instead.

    Example::

       cast([0.9, 1.3], dtype='int32') = [0, 1]
       cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]
       cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]



    Defined in ../src/operator/tensor/elemwise_unary_op_basic.cc:L664

    Parameters
    ----------
    data : NDArray
        The input.
    dtype : {'bfloat16', 'bool', 'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'}, required
        Output data type.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)