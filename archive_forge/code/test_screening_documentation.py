import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from statsmodels.discrete.discrete_model import Poisson, Logit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import family
from statsmodels.base._penalized import PenalizedMixin
from statsmodels.base._screening import VariableScreening

Created on Wed May 23 12:53:27 2018

Author: Josef Perktold

