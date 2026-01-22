from ._internal import NDArrayBase
from ..base import _Null
def SyncBatchNorm(data=None, gamma=None, beta=None, moving_mean=None, moving_var=None, eps=_Null, momentum=_Null, fix_gamma=_Null, use_global_stats=_Null, output_mean_var=_Null, ndev=_Null, key=_Null, out=None, name=None, **kwargs):
    """Batch normalization.

    Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
    well as offset ``beta``.
    Standard BN [1]_ implementation only normalize the data within each device.
    SyncBN normalizes the input within the whole mini-batch.
    We follow the sync-onece implmentation described in the paper [2]_.

    Assume the input has more than one dimension and we normalize along axis 1.
    We first compute the mean and variance along this axis:

    .. math::

      data\\_mean[i] = mean(data[:,i,:,...]) \\\\
      data\\_var[i] = var(data[:,i,:,...])

    Then compute the normalized output, which has the same shape as input, as following:

    .. math::

      out[:,i,:,...] = \\frac{data[:,i,:,...] - data\\_mean[i]}{\\sqrt{data\\_var[i]+\\epsilon}} * gamma[i] + beta[i]

    Both *mean* and *var* returns a scalar by treating the input as a vector.

    Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
    have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
    ``data_var`` as well, which are needed for the backward pass.

    Besides the inputs and the outputs, this operator accepts two auxiliary
    states, ``moving_mean`` and ``moving_var``, which are *k*-length
    vectors. They are global statistics for the whole dataset, which are updated
    by::

      moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
      moving_var = moving_var * momentum + data_var * (1 - momentum)

    If ``use_global_stats`` is set to be true, then ``moving_mean`` and
    ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
    the output. It is often used during inference.

    Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,
    then set ``gamma`` to 1 and its gradient to 0.

    Reference:
      .. [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating \\
        deep network training by reducing internal covariate shift." *ICML 2015*
      .. [2] Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, \\
        Ambrish Tyagi, and Amit Agrawal. "Context Encoding for Semantic Segmentation." *CVPR 2018*


    Defined in ../src/operator/contrib/sync_batch_norm.cc:L96

    Parameters
    ----------
    data : NDArray
        Input data to batch normalization
    gamma : NDArray
        gamma array
    beta : NDArray
        beta array
    moving_mean : NDArray
        running mean of input
    moving_var : NDArray
        running variance of input
    eps : float, optional, default=0.00100000005
        Epsilon to prevent div 0
    momentum : float, optional, default=0.899999976
        Momentum for moving average
    fix_gamma : boolean, optional, default=1
        Fix gamma while training
    use_global_stats : boolean, optional, default=0
        Whether use global moving statistics instead of local batch-norm. This will force change batch-norm into a scale shift operator.
    output_mean_var : boolean, optional, default=0
        Output All,normal mean and var
    ndev : int, optional, default='1'
        The count of GPU devices
    key : string, required
        Hash key for synchronization, please set the same hash key for same layer, Block.prefix is typically used as in :class:`gluon.nn.contrib.SyncBatchNorm`.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)