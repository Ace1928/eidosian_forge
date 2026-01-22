from ._internal import NDArrayBase
from ..base import _Null
def _onehot_encode(lhs=None, rhs=None, out=None, name=None, **kwargs):
    """

    Parameters
    ----------
    lhs : NDArray
        Left operand to the function.
    rhs : NDArray
        Right operand to the function.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)