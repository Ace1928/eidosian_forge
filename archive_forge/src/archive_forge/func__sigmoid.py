from __future__ import absolute_import, division, print_function
import numpy as np
def _sigmoid(x, out=None):
    """
        Logistic sigmoid function.

        Parameters
        ----------
        x : numpy array
            Input data.
        out : numpy array, optional
            Array to hold the output data.

        Returns
        -------
        numpy array
            Logistic sigmoid of input data.

        """
    if out is None:
        out = np.asarray(0.5 * x)
    else:
        if out is not x:
            out[:] = x
        out *= 0.5
    np.tanh(out, out=out)
    out += 1
    out *= 0.5
    return out