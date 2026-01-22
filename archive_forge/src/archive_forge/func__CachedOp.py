from ._internal import NDArrayBase
from ..base import _Null
def _CachedOp(*data, **kwargs):
    """

    Parameters
    ----------
    data : NDArray[]
        input data list

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)