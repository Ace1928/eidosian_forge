from ._internal import NDArrayBase
from ..base import _Null
def _backward_div_scalar(data=None, scalar=_Null, is_int=_Null, out=None, name=None, **kwargs):
    """

    Parameters
    ----------
    data : NDArray
        source input
    scalar : double, optional, default=1
        Scalar input value
    is_int : boolean, optional, default=1
        Indicate whether scalar input is int type

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)