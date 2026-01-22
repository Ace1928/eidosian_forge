import warnings
import functools
from .. import symbol, init, ndarray
from ..base import string_types, numeric_types
class ConvRNNCell(BaseConvRNNCell):
    """Convolutional RNN cells

    Parameters
    ----------
    input_shape : tuple of int
        Shape of input in single timestep.
    num_hidden : int
        Number of units in output symbol.
    h2h_kernel : tuple of int, default (3, 3)
        Kernel of Convolution operator in state-to-state transitions.
    h2h_dilate : tuple of int, default (1, 1)
        Dilation of Convolution operator in state-to-state transitions.
    i2h_kernel : tuple of int, default (3, 3)
        Kernel of Convolution operator in input-to-state transitions.
    i2h_stride : tuple of int, default (1, 1)
        Stride of Convolution operator in input-to-state transitions.
    i2h_pad : tuple of int, default (1, 1)
        Pad of Convolution operator in input-to-state transitions.
    i2h_dilate : tuple of int, default (1, 1)
        Dilation of Convolution operator in input-to-state transitions.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the convolution
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the convolution
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer, default zeros
        Initializer for the bias vector.
    h2h_bias_initializer : str or Initializer, default zeros
        Initializer for the bias vector.
    activation : str or Symbol,
        default functools.partial(symbol.LeakyReLU, act_type='leaky', slope=0.2)
        Type of activation function.
    prefix : str, default ``'ConvRNN_'``
        Prefix for name of layers (and name of weight if params is None).
    params : RNNParams, default None
        Container for weight sharing between cells. Created if None.
    conv_layout : str, , default 'NCHW'
        Layout of ConvolutionOp
    """

    def __init__(self, input_shape, num_hidden, h2h_kernel=(3, 3), h2h_dilate=(1, 1), i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1), i2h_dilate=(1, 1), i2h_weight_initializer=None, h2h_weight_initializer=None, i2h_bias_initializer='zeros', h2h_bias_initializer='zeros', activation=functools.partial(symbol.LeakyReLU, act_type='leaky', slope=0.2), prefix='ConvRNN_', params=None, conv_layout='NCHW'):
        super(ConvRNNCell, self).__init__(input_shape=input_shape, num_hidden=num_hidden, h2h_kernel=h2h_kernel, h2h_dilate=h2h_dilate, i2h_kernel=i2h_kernel, i2h_stride=i2h_stride, i2h_pad=i2h_pad, i2h_dilate=i2h_dilate, i2h_weight_initializer=i2h_weight_initializer, h2h_weight_initializer=h2h_weight_initializer, i2h_bias_initializer=i2h_bias_initializer, h2h_bias_initializer=h2h_bias_initializer, activation=activation, prefix=prefix, params=params, conv_layout=conv_layout)

    @property
    def _gate_names(self):
        return ('',)

    def __call__(self, inputs, states):
        self._counter += 1
        name = '%st%d_' % (self._prefix, self._counter)
        i2h, h2h = self._conv_forward(inputs, states, name)
        output = self._get_activation(i2h + h2h, self._activation, name='%sout' % name)
        return (output, [output])

    @property
    def state_info(self):
        return [{'shape': self._state_shape, '__layout__': self._conv_layout}]