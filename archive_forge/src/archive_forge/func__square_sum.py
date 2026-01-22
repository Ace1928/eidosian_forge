from ._internal import NDArrayBase
from ..base import _Null
def _square_sum(data=None, axis=_Null, keepdims=_Null, exclude=_Null, out=None, name=None, **kwargs):
    """Computes the square sum of array elements over a given axis
    for row-sparse matrix. This is a temporary solution for fusing ops square and
    sum together for row-sparse matrix to save memory for storing gradients.
    It will become deprecated once the functionality of fusing operators is finished
    in the future.

    Example::

      dns = mx.nd.array([[0, 0], [1, 2], [0, 0], [3, 4], [0, 0]])
      rsp = dns.tostype('row_sparse')
      sum = mx.nd._internal._square_sum(rsp, axis=1)
      sum = [0, 5, 0, 25, 0]


    Defined in ../src/operator/tensor/square_sum.cc:L63

    Parameters
    ----------
    data : NDArray
        The input
    axis : Shape or None, optional, default=None
        The axis or axes along which to perform the reduction.

          The default, `axis=()`, will compute over all elements into a
          scalar array with shape `(1,)`.

          If `axis` is int, a reduction is performed on a particular axis.

          If `axis` is a tuple of ints, a reduction is performed on all the axes
          specified in the tuple.

          If `exclude` is true, reduction will be performed on the axes that are
          NOT in axis instead.

          Negative values means indexing from right to left.
    keepdims : boolean, optional, default=0
        If this is set to `True`, the reduced axes are left in the result as dimension with size one.
    exclude : boolean, optional, default=0
        Whether to perform reduction on axis that are NOT in axis instead.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)