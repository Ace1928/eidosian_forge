from math import floor
from ....base import numeric_types
from ...rnn import HybridRecurrentCell
class Conv3DLSTMCell(_ConvLSTMCell):
    """3D Convolutional LSTM network cell.

    `"Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
    <https://arxiv.org/abs/1506.04214>`_ paper. Xingjian et al. NIPS2015

    .. math::
        \\begin{array}{ll}
        i_t = \\sigma(W_i \\ast x_t + R_i \\ast h_{t-1} + b_i) \\\\
        f_t = \\sigma(W_f \\ast x_t + R_f \\ast h_{t-1} + b_f) \\\\
        o_t = \\sigma(W_o \\ast x_t + R_o \\ast h_{t-1} + b_o) \\\\
        c^\\prime_t = tanh(W_c \\ast x_t + R_c \\ast h_{t-1} + b_c) \\\\
        c_t = f_t \\circ c_{t-1} + i_t \\circ c^\\prime_t \\\\
        h_t = o_t \\circ tanh(c_t) \\\\
        \\end{array}

    Parameters
    ----------
    input_shape : tuple of int
        Input tensor shape at each time step for each sample, excluding dimension of the batch size
        and sequence length. Must be consistent with `conv_layout`.
        For example, for layout 'NCDHW' the shape should be (C, D, H, W).
    hidden_channels : int
        Number of output channels.
    i2h_kernel : int or tuple of int
        Input convolution kernel sizes.
    h2h_kernel : int or tuple of int
        Recurrent convolution kernel sizes. Only odd-numbered sizes are supported.
    i2h_pad : int or tuple of int, default (0, 0, 0)
        Pad for input convolution.
    i2h_dilate : int or tuple of int, default (1, 1, 1)
        Input convolution dilate.
    h2h_dilate : int or tuple of int, default (1, 1, 1)
        Recurrent convolution dilate.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the input convolutions.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the input convolutions.
    i2h_bias_initializer : str or Initializer, default zeros
        Initializer for the input convolution bias vectors.
    h2h_bias_initializer : str or Initializer, default zeros
        Initializer for the recurrent convolution bias vectors.
    conv_layout : str, default 'NCDHW'
        Layout for all convolution inputs, outputs and weights. Options are 'NCDHW' and 'NDHWC'.
    activation : str or gluon.Block, default 'tanh'
        Type of activation function used in c^\\prime_t.
        If argument type is string, it's equivalent to nn.Activation(act_type=str). See
        :func:`~mxnet.ndarray.Activation` for available choices.
        Alternatively, other activation blocks such as nn.LeakyReLU can be used.
    prefix : str, default ``'conv_lstm_``'
        Prefix for name of layers (and name of weight if params is None).
    params : RNNParams, default None
        Container for weight sharing between cells. Created if None.
    """

    def __init__(self, input_shape, hidden_channels, i2h_kernel, h2h_kernel, i2h_pad=(0, 0, 0), i2h_dilate=(1, 1, 1), h2h_dilate=(1, 1, 1), i2h_weight_initializer=None, h2h_weight_initializer=None, i2h_bias_initializer='zeros', h2h_bias_initializer='zeros', conv_layout='NCDHW', activation='tanh', prefix=None, params=None):
        super(Conv3DLSTMCell, self).__init__(input_shape=input_shape, hidden_channels=hidden_channels, i2h_kernel=i2h_kernel, h2h_kernel=h2h_kernel, i2h_pad=i2h_pad, i2h_dilate=i2h_dilate, h2h_dilate=h2h_dilate, i2h_weight_initializer=i2h_weight_initializer, h2h_weight_initializer=h2h_weight_initializer, i2h_bias_initializer=i2h_bias_initializer, h2h_bias_initializer=h2h_bias_initializer, dims=3, conv_layout=conv_layout, activation=activation, prefix=prefix, params=params)