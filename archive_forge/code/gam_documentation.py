import numpy as np
from statsmodels.genmod import families
from statsmodels.sandbox.nonparametric.smoothers import PolySmoother
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import IterationLimitWarning, iteration_limit_doc
import warnings

        Return Pearson's X^2 estimate of scale.
        