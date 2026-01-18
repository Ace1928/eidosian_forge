import numpy as np
import warnings
from scipy import stats, optimize
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.stats._inference_tools import _mover_confint
waldccv