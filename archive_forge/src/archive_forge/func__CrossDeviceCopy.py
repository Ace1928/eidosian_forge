from ._internal import NDArrayBase
from ..base import _Null
def _CrossDeviceCopy(out=None, name=None, **kwargs):
    """Special op to copy data cross device

    Parameters
    ----------


    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)