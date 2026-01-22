import numpy as np
import pandas as pd
import os, os.path
from sklearn.base import BaseEstimator

        Get the boolean mask indicating which features are selected
        Returns
        -------
        support : boolean array of shape [# input features]
            An element is True iff its corresponding feature is selected for
            retention.
        