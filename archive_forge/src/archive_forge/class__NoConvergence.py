import collections
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize._optimize import _check_unknown_options
from ._linesearch import _nonmonotone_line_search_cruz, _nonmonotone_line_search_cheng
class _NoConvergence(Exception):
    pass