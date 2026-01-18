import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, \

        Calculates the expected conditional mean
        m(X, Z=l) for all possible l
        