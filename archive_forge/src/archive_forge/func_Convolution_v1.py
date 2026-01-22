from ._internal import NDArrayBase
from ..base import _Null
def Convolution_v1(data=None, weight=None, bias=None, kernel=_Null, stride=_Null, dilate=_Null, pad=_Null, num_filter=_Null, num_group=_Null, workspace=_Null, no_bias=_Null, cudnn_tune=_Null, cudnn_off=_Null, layout=_Null, out=None, name=None, **kwargs):
    """This operator is DEPRECATED. Apply convolution to input then add a bias.

    Parameters
    ----------
    data : NDArray
        Input data to the ConvolutionV1Op.
    weight : NDArray
        Weight matrix.
    bias : NDArray
        Bias parameter.
    kernel : Shape(tuple), required
        convolution kernel size: (h, w) or (d, h, w)
    stride : Shape(tuple), optional, default=[]
        convolution stride: (h, w) or (d, h, w)
    dilate : Shape(tuple), optional, default=[]
        convolution dilate: (h, w) or (d, h, w)
    pad : Shape(tuple), optional, default=[]
        pad for convolution: (h, w) or (d, h, w)
    num_filter : int (non-negative), required
        convolution filter(channel) number
    num_group : int (non-negative), optional, default=1
        Number of group partitions. Equivalent to slicing input into num_group
        partitions, apply convolution on each, then concatenate the results
    workspace : long (non-negative), optional, default=1024
        Maximum temporary workspace allowed for convolution (MB).This parameter determines the effective batch size of the convolution kernel, which may be smaller than the given batch size. Also, the workspace will be automatically enlarged to make sure that we can run the kernel with batch_size=1
    no_bias : boolean, optional, default=0
        Whether to disable bias parameter.
    cudnn_tune : {None, 'fastest', 'limited_workspace', 'off'},optional, default='None'
        Whether to pick convolution algo by running performance test.
        Leads to higher startup time but may give faster speed. Options are:
        'off': no tuning
        'limited_workspace': run test and pick the fastest algorithm that doesn't exceed workspace limit.
        'fastest': pick the fastest algorithm and ignore workspace limit.
        If set to None (default), behavior is determined by environment
        variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,
        1 for limited workspace (default), 2 for fastest.
    cudnn_off : boolean, optional, default=0
        Turn off cudnn for this layer.
    layout : {None, 'NCDHW', 'NCHW', 'NDHWC', 'NHWC'},optional, default='None'
        Set layout for input, output and weight. Empty for
        default layout: NCHW for 2d and NCDHW for 3d.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)