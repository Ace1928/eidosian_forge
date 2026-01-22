from ._internal import NDArrayBase
from ..base import _Null
def _multi_mp_adamw_update(*data, **kwargs):
    """Update function for multi-precision AdamW optimizer.

    AdamW is seen as a modification of Adam by decoupling the weight decay from the
    optimization steps taken w.r.t. the loss function.

    Adam update consists of the following steps, where g represents gradient and m, v
    are 1st and 2nd order moment estimates (mean and variance).

    .. math::

     g_t = \\nabla J(W_{t-1})\\\\
     m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t\\\\
     v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2\\\\
     W_t = W_{t-1} - \\eta_t (\\alpha \\frac{ m_t }{ \\sqrt{ v_t } + \\epsilon } + wd W_{t-1})

    It updates the weights using::

     m = beta1*m + (1-beta1)*grad
     v = beta2*v + (1-beta2)*(grad**2)
     w -= eta * (learning_rate * m / (sqrt(v) + epsilon) + w * wd)

    Note that gradient is rescaled to grad = rescale_grad * grad. If rescale_grad is NaN, Inf, or 0,
    the update is skipped.


    Defined in ../src/operator/contrib/adamw.cc:L222

    Parameters
    ----------
    data : NDArray[]
        data
    lrs : tuple of <float>, required
        Learning rates
    beta1 : float, optional, default=0.899999976
        The decay rate for the 1st moment estimates.
    beta2 : float, optional, default=0.999000013
        The decay rate for the 2nd moment estimates.
    epsilon : float, optional, default=9.99999994e-09
        A small constant for numerical stability.
    wds : tuple of <float>, required
        Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
    etas : tuple of <float>, required
        Learning rates schedule multiplier
    clip_gradient : float, optional, default=-1
        Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
    num_weights : int, optional, default='1'
        Number of updated weights.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)