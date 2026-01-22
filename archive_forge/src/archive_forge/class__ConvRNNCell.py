from math import floor
from ....base import numeric_types
from ...rnn import HybridRecurrentCell
class _ConvRNNCell(_BaseConvRNNCell):

    def __init__(self, input_shape, hidden_channels, i2h_kernel, h2h_kernel, i2h_pad, i2h_dilate, h2h_dilate, i2h_weight_initializer, h2h_weight_initializer, i2h_bias_initializer, h2h_bias_initializer, dims, conv_layout, activation, prefix, params):
        super(_ConvRNNCell, self).__init__(input_shape=input_shape, hidden_channels=hidden_channels, activation=activation, i2h_kernel=i2h_kernel, i2h_pad=i2h_pad, i2h_dilate=i2h_dilate, h2h_kernel=h2h_kernel, h2h_dilate=h2h_dilate, i2h_weight_initializer=i2h_weight_initializer, h2h_weight_initializer=h2h_weight_initializer, i2h_bias_initializer=i2h_bias_initializer, h2h_bias_initializer=h2h_bias_initializer, dims=dims, conv_layout=conv_layout, prefix=prefix, params=params)

    def state_info(self, batch_size=0):
        return [{'shape': (batch_size,) + self._state_shape, '__layout__': self._conv_layout}]

    def _alias(self):
        return 'conv_rnn'

    @property
    def _gate_names(self):
        return ('',)

    def hybrid_forward(self, F, inputs, states, i2h_weight, h2h_weight, i2h_bias, h2h_bias):
        prefix = 't%d_' % self._counter
        i2h, h2h = self._conv_forward(F, inputs, states, i2h_weight, h2h_weight, i2h_bias, h2h_bias, prefix)
        output = self._get_activation(F, i2h + h2h, self._activation, name=prefix + 'out')
        return (output, [output])