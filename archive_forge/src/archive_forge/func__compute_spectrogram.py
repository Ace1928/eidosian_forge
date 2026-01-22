import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn import preprocessing, decomposition
import scprep
from . import utils
def _compute_spectrogram(self, sample_indicator, window):
    """Computes spectrograms for arbitrary window/signal/graph combinations

        Parameters
        ----------
        sample_indicator : np.ndarray
            Input signal
        U : np.ndarray
            eigenvectors
        window : TYPE
            window matrix

        Returns
        -------
        C
            Normalized Spectrogram

        Raises
        ------
        TypeError
            Description
        """
    if len(sample_indicator.shape) == 1:
        sample_indicator = np.array(sample_indicator)
    else:
        raise ValueError('sample_indicator must be 1-dimensional. Got shape: {}'.format(sample_indicator.shape))
    if sparse.issparse(window):
        C = window.multiply(sample_indicator).toarray()
    else:
        C = np.multiply(window, sample_indicator)
    C = preprocessing.normalize(self.eigenvectors.T @ C, axis=0)
    return C.T