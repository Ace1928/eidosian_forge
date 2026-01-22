from ._internal import NDArrayBase
from ..base import _Null
def _multi_mp_lamb_update(*data, **kwargs):
    """Compute the LAMB coefficients of multiple weights and grads with Mix Precision"


    Defined in ../src/operator/contrib/multi_lamb.cc:L213

    Parameters
    ----------
    data : NDArray[]
        data
    learning_rates : tuple of <float>, required
        List of learning rates
    beta1 : float, optional, default=0.899999976
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional, default=0.999000013
        Exponential decay rate for the second moment estimates.
    epsilon : float, optional, default=9.99999997e-07
        Small value to avoid division by 0.
    wds : tuple of <float>, required
        List of Weight decays.Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
    rescale_grad : float, optional, default=1
        Gradient rescaling factor
    lower_bound : float, optional, default=-1
        Lower limit of norm of weight. If lower_bound <= 0, Lower limit is not set
    upper_bound : float, optional, default=-1
        Upper limit of norm of weight. If upper_bound <= 0, Upper limit is not set
    clip_gradient : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
    bias_correction : boolean, optional, default=1
        Whether to use bias correction.
    step_count : Shape(tuple), required
        Step count for each tensor
    num_tensors : int, optional, default='1'
        Number of tensors

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)