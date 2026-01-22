from __future__ import absolute_import, division, print_function
import numpy as np
from .activations import linear, sigmoid, tanh
class RecurrentLayer(FeedForwardLayer):
    """
    Recurrent network layer.

    Parameters
    ----------
    weights : numpy array, shape (num_inputs, num_hiddens)
        Weights.
    bias : scalar or numpy array, shape (num_hiddens,)
        Bias.
    recurrent_weights : numpy array, shape (num_hiddens, num_hiddens)
        Recurrent weights.
    activation_fn : numpy ufunc
        Activation function.
    init : numpy array, shape (num_hiddens,), optional
        Initial state of hidden units.

    """

    def __init__(self, weights, bias, recurrent_weights, activation_fn, init=None):
        super(RecurrentLayer, self).__init__(weights, bias, activation_fn)
        self.recurrent_weights = recurrent_weights
        if init is None:
            init = np.zeros(self.bias.size, dtype=NN_DTYPE)
        self.init = init
        self._prev = self.init

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_prev', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'init'):
            self.init = np.zeros(self.bias.size, dtype=NN_DTYPE)
        self._prev = self.init

    def reset(self, init=None):
        """
        Reset the layer to its initial state.

        Parameters
        ----------
        init : numpy array, shape (num_hiddens,), optional
            Reset the hidden units to this initial state.

        """
        self._prev = init if init is not None else self.init

    def activate(self, data, reset=True):
        """
        Activate the layer.

        Parameters
        ----------
        data : numpy array, shape (num_frames, num_inputs)
            Activate with this data.
        reset : bool, optional
            Reset the layer to its initial state before activating it.

        Returns
        -------
        numpy array, shape (num_frames, num_hiddens)
            Activations for this data.

        """
        if reset:
            self.reset()
        out = np.dot(data, self.weights) + self.bias
        for i in range(len(data)):
            out[i] += np.dot(self._prev, self.recurrent_weights)
            out[i] = self.activation_fn(out[i])
            self._prev = out[i]
        return out