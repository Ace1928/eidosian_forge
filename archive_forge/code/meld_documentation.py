import numpy as np
import pandas as pd
import graphtools
from . import utils
from . import filter
from graphtools.estimator import GraphEstimator, attribute
from functools import partial
Builds the MELD filter over a graph built on data `X` and estimates density
        of each sample in `sample_labels`

        Parameters
        ----------

        X : array-like, shape=[n_samples, m_features]
            Data on which to build graph to perform data smoothing over.

        sample_labels : array-like, shape=[n_samples, p_signals]
            1- or 2-dimensional array of non-numerics indicating the sample origin for
            each cell.

        kwargs : additional arguments for graphtools.Graph

        Returns
        -------
        sample_densities : ndarray, shape=[n_samples, p_signals]
            Density estimate for each sample over a graph built from X
        