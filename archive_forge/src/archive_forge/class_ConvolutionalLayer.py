from __future__ import absolute_import, division, print_function
import numpy as np
from .activations import linear, sigmoid, tanh
class ConvolutionalLayer(FeedForwardLayer):
    """
    Convolutional network layer.

    Parameters
    ----------
    weights : numpy array, shape (num_feature_maps, num_channels, <kernel>)
        Weights.
    bias : scalar or numpy array, shape (num_filters,)
        Bias.
    stride : int, optional
        Stride of the convolution.
    pad : {'valid', 'same', 'full'}
        A string indicating the size of the output:

        - full
            The output is the full discrete linear convolution of the inputs.
        - valid
            The output consists only of those elements that do not rely on the
            zero-padding.
        - same
            The output is the same size as the input, centered with respect to
            the ‘full’ output.

    activation_fn : numpy ufunc
        Activation function.

    """

    def __init__(self, weights, bias, stride=1, pad='valid', activation_fn=linear):
        super(ConvolutionalLayer, self).__init__(weights, bias, activation_fn)
        if stride != 1:
            raise NotImplementedError('only `stride` == 1 implemented.')
        self.stride = stride
        if pad != 'valid':
            raise NotImplementedError('only `pad` == "valid" implemented.')
        self.pad = pad

    def activate(self, data, **kwargs):
        """
        Activate the layer.

        Parameters
        ----------
        data : numpy array (num_frames, num_bins, num_channels)
            Activate with this data.

        Returns
        -------
        numpy array
            Activations for this data.

        """
        if len(data.shape) == 2:
            data = data.reshape(data.shape + (1,))
        num_frames, num_bins, num_channels = data.shape
        num_channels_w, num_features, size_time, size_freq = self.weights.shape
        if num_channels_w != num_channels:
            raise ValueError('Number of channels in weight vector different from number of channels of input data!')
        num_frames -= size_time - 1
        num_bins -= size_freq - 1
        out = np.zeros((num_frames, num_bins, num_features), dtype=NN_DTYPE, order='F')
        for c in range(num_channels):
            channel = data[:, :, c]
            for w, weights in enumerate(self.weights[c]):
                conv = convolve(channel, weights)
                out[:, :, w] += conv
        return self.activation_fn(out + self.bias)