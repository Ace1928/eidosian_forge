from ._internal import NDArrayBase
from ..base import _Null
def GroupNorm(data=None, gamma=None, beta=None, num_groups=_Null, eps=_Null, output_mean_var=_Null, out=None, name=None, **kwargs):
    """Group normalization.

    The input channels are separated into ``num_groups`` groups, each containing ``num_channels / num_groups`` channels.
    The mean and standard-deviation are calculated separately over the each group.

    .. math::

      data = data.reshape((N, num_groups, C // num_groups, ...))
      out = \\frac{data - mean(data, axis)}{\\sqrt{var(data, axis) + \\epsilon}} * gamma + beta

    Both ``gamma`` and ``beta`` are learnable parameters.



    Defined in ../src/operator/nn/group_norm.cc:L76

    Parameters
    ----------
    data : NDArray
        Input data
    gamma : NDArray
        gamma array
    beta : NDArray
        beta array
    num_groups : int, optional, default='1'
        Total number of groups.
    eps : float, optional, default=9.99999975e-06
        An `epsilon` parameter to prevent division by 0.
    output_mean_var : boolean, optional, default=0
        Output the mean and std calculated along the given axis.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)