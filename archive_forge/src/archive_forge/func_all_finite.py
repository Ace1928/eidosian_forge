from ._internal import NDArrayBase
from ..base import _Null
def all_finite(data=None, init_output=_Null, out=None, name=None, **kwargs):
    """Check if all the float numbers in the array are finite (used for AMP)


    Defined in ../src/operator/contrib/all_finite.cc:L100

    Parameters
    ----------
    data : NDArray
        Array
    init_output : boolean, optional, default=1
        Initialize output to 1.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)