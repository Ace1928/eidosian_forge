from ._internal import NDArrayBase
from ..base import _Null
def _backward_dot(transpose_a=_Null, transpose_b=_Null, forward_stype=_Null, out=None, name=None, **kwargs):
    """

    Parameters
    ----------
    transpose_a : boolean, optional, default=0
        If true then transpose the first input before dot.
    transpose_b : boolean, optional, default=0
        If true then transpose the second input before dot.
    forward_stype : {None, 'csr', 'default', 'row_sparse'},optional, default='None'
        The desired storage type of the forward output given by user, if thecombination of input storage types and this hint does not matchany implemented ones, the dot operator will perform fallback operationand still produce an output of the desired storage type.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)