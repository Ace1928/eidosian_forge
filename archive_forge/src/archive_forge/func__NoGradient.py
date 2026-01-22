from ._internal import NDArrayBase
from ..base import _Null
def _NoGradient(out=None, name=None, **kwargs):
    """Place holder for variable who cannot perform gradient

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