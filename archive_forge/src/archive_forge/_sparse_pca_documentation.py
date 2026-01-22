from numbers import Integral, Real
import numpy as np
from ..base import (
from ..linear_model import ridge_regression
from ..utils import check_random_state
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.extmath import svd_flip
from ..utils.validation import check_array, check_is_fitted
from ._dict_learning import MiniBatchDictionaryLearning, dict_learning
Specialized `fit` for MiniBatchSparsePCA.