from ._internal import NDArrayBase
from ..base import _Null
def _backward_arctanh(lhs=None, rhs=None, out=None, name=None, **kwargs):
    """

    Parameters
    ----------
    lhs : NDArray
        first input
    rhs : NDArray
        second input

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)