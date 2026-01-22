from .... import symbol
from ...block import HybridBlock
from ....base import numeric_types
from ...nn import Activation
2-D Deformable Convolution v2 (Dai, 2018).

    The modulated deformable convolution operation is described in https://arxiv.org/abs/1811.11168

    Parameters
    ----------
    channels : int,
        The dimensionality of the output space
        i.e. the number of output channels in the convolution.
    kernel_size : int or tuple/list of 2 ints, (Default value = (1,1))
        Specifies the dimensions of the convolution window.
    strides : int or tuple/list of 2 ints, (Default value = (1,1))
        Specifies the strides of the convolution.
    padding : int or tuple/list of 2 ints, (Default value = (0,0))
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points.
    dilation : int or tuple/list of 2 ints, (Default value = (1,1))
        Specifies the dilation rate to use for dilated convolution.
    groups : int, (Default value = 1)
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two convolution
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    num_deformable_group : int, (Default value = 1)
        Number of deformable group partitions.
    layout : str, (Default value = NCHW)
        Dimension ordering of data and weight. Can be 'NCW', 'NWC', 'NCHW',
        'NHWC', 'NCDHW', 'NDHWC', etc. 'N', 'C', 'H', 'W', 'D' stands for
        batch, channel, height, width and depth dimensions respectively.
        Convolution is performed over 'D', 'H', and 'W' dimensions.
    use_bias : bool, (Default value = True)
        Whether the layer for generating the output features uses a bias vector.
    in_channels : int, (Default value = 0)
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and input channels will be inferred from the shape of input data.
    activation : str, (Default value = None)
        Activation function to use. See :func:`~mxnet.ndarray.Activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    weight_initializer : str or `Initializer`, (Default value = None)
        Initializer for the `weight` weights matrix for the convolution layer
        for generating the output features.
    bias_initializer : str or `Initializer`, (Default value = zeros)
        Initializer for the bias vector for the convolution layer
        for generating the output features.
    offset_weight_initializer : str or `Initializer`, (Default value = zeros)
        Initializer for the `weight` weights matrix for the convolution layer
        for generating the offset.
    offset_bias_initializer : str or `Initializer`, (Default value = zeros),
        Initializer for the bias vector for the convolution layer
        for generating the offset.
    offset_use_bias: bool, (Default value = True)
        Whether the layer for generating the offset uses a bias vector.

    Inputs:
        - **data**: 4D input tensor with shape
          `(batch_size, in_channels, height, width)` when `layout` is `NCHW`.
          For other layouts shape is permuted accordingly.

    Outputs:
        - **out**: 4D output tensor with shape
          `(batch_size, channels, out_height, out_width)` when `layout` is `NCHW`.
          out_height and out_width are calculated as::

              out_height = floor((height+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1
              out_width = floor((width+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1
    