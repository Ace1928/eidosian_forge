from ._internal import NDArrayBase
from ..base import _Null
def ModulatedDeformableConvolution(data=None, offset=None, mask=None, weight=None, bias=None, kernel=_Null, stride=_Null, dilate=_Null, pad=_Null, num_filter=_Null, num_group=_Null, num_deformable_group=_Null, workspace=_Null, no_bias=_Null, im2col_step=_Null, layout=_Null, out=None, name=None, **kwargs):
    """Compute 2-D modulated deformable convolution on 4-D input.

    The modulated deformable convolution operation is described in https://arxiv.org/abs/1811.11168

    For 2-D modulated deformable convolution, the shapes are

    - **data**: *(batch_size, channel, height, width)*
    - **offset**: *(batch_size, num_deformable_group * kernel[0] * kernel[1] * 2, height, width)*
    - **mask**: *(batch_size, num_deformable_group * kernel[0] * kernel[1], height, width)*
    - **weight**: *(num_filter, channel, kernel[0], kernel[1])*
    - **bias**: *(num_filter,)*
    - **out**: *(batch_size, num_filter, out_height, out_width)*.

    Define::

      f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1

    then we have::

      out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
      out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])

    If ``no_bias`` is set to be true, then the ``bias`` term is ignored.

    The default data ``layout`` is *NCHW*, namely *(batch_size, channle, height,
    width)*.

    If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``
    evenly into *g* parts along the channel axis, and also evenly split ``weight``
    along the first dimension. Next compute the convolution on the *i*-th part of
    the data with the *i*-th weight part. The output is obtained by concating all
    the *g* results.

    If ``num_deformable_group`` is larger than 1, denoted by *dg*, then split the
    input ``offset`` evenly into *dg* parts along the channel axis, and also evenly
    split ``out`` evenly into *dg* parts along the channel axis. Next compute the
    deformable convolution, apply the *i*-th part of the offset part on the *i*-th
    out.


    Both ``weight`` and ``bias`` are learnable parameters.




    Defined in ../src/operator/contrib/modulated_deformable_convolution.cc:L83

    Parameters
    ----------
    data : NDArray
        Input data to the ModulatedDeformableConvolutionOp.
    offset : NDArray
        Input offset to ModulatedDeformableConvolutionOp.
    mask : NDArray
        Input mask to the ModulatedDeformableConvolutionOp.
    weight : NDArray
        Weight matrix.
    bias : NDArray
        Bias parameter.
    kernel : Shape(tuple), required
        Convolution kernel size: (h, w) or (d, h, w)
    stride : Shape(tuple), optional, default=[]
        Convolution stride: (h, w) or (d, h, w). Defaults to 1 for each dimension.
    dilate : Shape(tuple), optional, default=[]
        Convolution dilate: (h, w) or (d, h, w). Defaults to 1 for each dimension.
    pad : Shape(tuple), optional, default=[]
        Zero pad for convolution: (h, w) or (d, h, w). Defaults to no padding.
    num_filter : int (non-negative), required
        Convolution filter(channel) number
    num_group : int (non-negative), optional, default=1
        Number of group partitions.
    num_deformable_group : int (non-negative), optional, default=1
        Number of deformable group partitions.
    workspace : long (non-negative), optional, default=1024
        Maximum temperal workspace allowed for convolution (MB).
    no_bias : boolean, optional, default=0
        Whether to disable bias parameter.
    im2col_step : int (non-negative), optional, default=64
        Maximum number of images per im2col computation; The total batch size should be divisable by this value or smaller than this value; if you face out of memory problem, you can try to use a smaller value here.
    layout : {None, 'NCDHW', 'NCHW', 'NCW'},optional, default='None'
        Set layout for input, output and weight. Empty for
        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)