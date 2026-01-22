import warnings
import functools
from .. import symbol, init, ndarray
from ..base import string_types, numeric_types
class BaseConvRNNCell(BaseRNNCell):
    """Abstract base class for Convolutional RNN cells"""

    def __init__(self, input_shape, num_hidden, h2h_kernel, h2h_dilate, i2h_kernel, i2h_stride, i2h_pad, i2h_dilate, i2h_weight_initializer, h2h_weight_initializer, i2h_bias_initializer, h2h_bias_initializer, activation, prefix='', params=None, conv_layout='NCHW'):
        super(BaseConvRNNCell, self).__init__(prefix=prefix, params=params)
        self._h2h_kernel = h2h_kernel
        assert self._h2h_kernel[0] % 2 == 1 and self._h2h_kernel[1] % 2 == 1, 'Only support odd number, get h2h_kernel= %s' % str(h2h_kernel)
        self._h2h_pad = (h2h_dilate[0] * (h2h_kernel[0] - 1) // 2, h2h_dilate[1] * (h2h_kernel[1] - 1) // 2)
        self._h2h_dilate = h2h_dilate
        self._i2h_kernel = i2h_kernel
        self._i2h_stride = i2h_stride
        self._i2h_pad = i2h_pad
        self._i2h_dilate = i2h_dilate
        self._num_hidden = num_hidden
        self._input_shape = input_shape
        self._conv_layout = conv_layout
        self._activation = activation
        data = symbol.Variable('data')
        self._state_shape = symbol.Convolution(data=data, num_filter=self._num_hidden, kernel=self._i2h_kernel, stride=self._i2h_stride, pad=self._i2h_pad, dilate=self._i2h_dilate, layout=conv_layout)
        self._state_shape = self._state_shape.infer_shape(data=input_shape)[1][0]
        self._state_shape = (0,) + self._state_shape[1:]
        self._iW = self.params.get('i2h_weight', init=i2h_weight_initializer)
        self._hW = self.params.get('h2h_weight', init=h2h_weight_initializer)
        self._iB = self.params.get('i2h_bias', init=i2h_bias_initializer)
        self._hB = self.params.get('h2h_bias', init=h2h_bias_initializer)

    @property
    def _num_gates(self):
        return len(self._gate_names)

    @property
    def state_info(self):
        return [{'shape': self._state_shape, '__layout__': self._conv_layout}, {'shape': self._state_shape, '__layout__': self._conv_layout}]

    def _conv_forward(self, inputs, states, name):
        i2h = symbol.Convolution(name='%si2h' % name, data=inputs, num_filter=self._num_hidden * self._num_gates, kernel=self._i2h_kernel, stride=self._i2h_stride, pad=self._i2h_pad, dilate=self._i2h_dilate, weight=self._iW, bias=self._iB, layout=self._conv_layout)
        h2h = symbol.Convolution(name='%sh2h' % name, data=states[0], num_filter=self._num_hidden * self._num_gates, kernel=self._h2h_kernel, dilate=self._h2h_dilate, pad=self._h2h_pad, stride=(1, 1), weight=self._hW, bias=self._hB, layout=self._conv_layout)
        return (i2h, h2h)

    def __call__(self, inputs, states):
        raise NotImplementedError('BaseConvRNNCell is abstract class for convolutional RNN')