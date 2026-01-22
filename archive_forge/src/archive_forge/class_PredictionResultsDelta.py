import numpy as np
from scipy import stats
import pandas as pd
class PredictionResultsDelta(PredictionResultsBase):
    """Prediction results based on delta method
    """

    def __init__(self, results_delta, **kwds):
        predicted = results_delta.predicted()
        var_pred = results_delta.var()
        super().__init__(predicted, var_pred, **kwds)