from __future__ import absolute_import, division, print_function
import numpy as np
from .activations import linear, sigmoid, tanh
class PadLayer(Layer):
    """
    Padding layer that pads the input with a constant value.

    Parameters
    ----------
    width : int
        Width of the padding (only one value for all dimensions)
    axes : iterable
        Indices of axes to be padded
    value : float
        Value to be used for padding.

    """

    def __init__(self, width, axes, value=0.0):
        self.width = width
        self.axes = axes
        self.value = value

    def activate(self, data, **kwargs):
        """
        Activate the layer.

        Parameters
        ----------
        data : numpy array
            Activate with this data.

        Returns
        -------
        numpy array
            Padded data.

        """
        shape = list(data.shape)
        data_idxs = [slice(None) for _ in range(len(shape))]
        for a in self.axes:
            shape[a] += self.width * 2
            data_idxs[a] = slice(self.width, -self.width)
        data_padded = np.full(tuple(shape), self.value)
        data_padded[tuple(data_idxs)] = data
        return data_padded