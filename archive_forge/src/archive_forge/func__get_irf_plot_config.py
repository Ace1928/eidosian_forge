from statsmodels.compat.python import lrange
import numpy as np
import statsmodels.tsa.vector_ar.util as util
def _get_irf_plot_config(names, impcol, rescol):
    nrows = ncols = k = len(names)
    if impcol is not None and rescol is not None:
        nrows = ncols = 1
        j = util.get_index(names, impcol)
        i = util.get_index(names, rescol)
        to_plot = [(j, i, 0, 0)]
    elif impcol is not None:
        ncols = 1
        j = util.get_index(names, impcol)
        to_plot = [(j, i, i, 0) for i in range(k)]
    elif rescol is not None:
        ncols = 1
        i = util.get_index(names, rescol)
        to_plot = [(j, i, j, 0) for j in range(k)]
    else:
        to_plot = [(j, i, i, j) for i in range(k) for j in range(k)]
    return (nrows, ncols, to_plot)