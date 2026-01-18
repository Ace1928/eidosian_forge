from . import utils
from scipy import sparse
from sklearn import decomposition
from sklearn import random_projection
import numpy as np
import pandas as pd
import sklearn.base
import warnings
Transform data back to its original space.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_components)

        Returns
        -------
        X_new : array-like, shape=(n_samples, n_features)
        