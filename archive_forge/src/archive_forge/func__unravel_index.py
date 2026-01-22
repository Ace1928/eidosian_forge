from ._internal import NDArrayBase
from ..base import _Null
def _unravel_index(data=None, shape=_Null, out=None, name=None, **kwargs):
    """Converts an array of flat indices into a batch of index arrays. The operator follows numpy conventions so a single multi index is given by a column of the output matrix. The leading dimension may be left unspecified by using -1 as placeholder.  

    Examples::

       A = [22,41,37]
       unravel(A, shape=(7,6)) = [[3,6,6],[4,5,1]]
       unravel(A, shape=(-1,6)) = [[3,6,6],[4,5,1]]



    Defined in ../src/operator/tensor/ravel.cc:L67

    Parameters
    ----------
    data : NDArray
        Array of flat indices
    shape : Shape(tuple), optional, default=None
        Shape of the array into which the multi-indices apply.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)