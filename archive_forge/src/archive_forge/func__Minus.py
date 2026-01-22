from ._internal import NDArrayBase
from ..base import _Null
def _Minus(lhs=None, rhs=None, out=None, name=None, **kwargs):
    """Subtracts arguments element-wise.

    The storage type of ``elemwise_sub`` output depends on storage types of inputs

       - elemwise_sub(row_sparse, row_sparse) = row_sparse
       - elemwise_sub(csr, csr) = csr
       - elemwise_sub(default, csr) = default
       - elemwise_sub(csr, default) = default
       - elemwise_sub(default, rsp) = default
       - elemwise_sub(rsp, default) = default
       - otherwise, ``elemwise_sub`` generates output with default storage



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