import pandas as pd
import numpy as np
import patsy
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
class PatsyFormula:
    """
    A simple wrapper for a string to be interpreted as a Patsy formula.
    """

    def __init__(self, formula):
        self.formula = '0 + ' + formula