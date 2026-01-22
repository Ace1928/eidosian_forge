from ._internal import NDArrayBase
from ..base import _Null
def SwapAxis(data=None, dim1=_Null, dim2=_Null, out=None, name=None, **kwargs):
    """Interchanges two axes of an array.

    Examples::

      x = [[1, 2, 3]])
      swapaxes(x, 0, 1) = [[ 1],
                           [ 2],
                           [ 3]]

      x = [[[ 0, 1],
            [ 2, 3]],
           [[ 4, 5],
            [ 6, 7]]]  // (2,2,2) array

     swapaxes(x, 0, 2) = [[[ 0, 4],
                           [ 2, 6]],
                          [[ 1, 5],
                           [ 3, 7]]]


    Defined in ../src/operator/swapaxis.cc:L69

    Parameters
    ----------
    data : NDArray
        Input array.
    dim1 : int, optional, default='0'
        the first axis to be swapped.
    dim2 : int, optional, default='0'
        the second axis to be swapped.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)