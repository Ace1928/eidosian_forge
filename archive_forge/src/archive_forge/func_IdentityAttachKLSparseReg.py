from ._internal import NDArrayBase
from ..base import _Null
def IdentityAttachKLSparseReg(data=None, sparseness_target=_Null, penalty=_Null, momentum=_Null, out=None, name=None, **kwargs):
    """Apply a sparse regularization to the output a sigmoid activation function.

    Parameters
    ----------
    data : NDArray
        Input data.
    sparseness_target : float, optional, default=0.100000001
        The sparseness target
    penalty : float, optional, default=0.00100000005
        The tradeoff parameter for the sparseness penalty
    momentum : float, optional, default=0.899999976
        The momentum for running average

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)