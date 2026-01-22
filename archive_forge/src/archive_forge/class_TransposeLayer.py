from __future__ import absolute_import, division, print_function
import numpy as np
from .activations import linear, sigmoid, tanh
class TransposeLayer(Layer):
    """
    Transpose layer.

    Parameters
    ----------
    axes : list of ints, optional
        By default, reverse the dimensions of the input, otherwise permute the
        axes of the input according to the values given.

    """

    def __init__(self, axes=None):
        self.axes = axes

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
            Transposed data.

        """
        return np.transpose(data, self.axes)