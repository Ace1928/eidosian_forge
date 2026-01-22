from collections import defaultdict
import numpy as np
from statsmodels.base._penalties import SCADSmoothed
class ScreeningResults:
    """Results for Variable Screening

    Note: Indices except for exog_idx and in the iterated case also
    idx_nonzero_batches are based on the combined [exog_keep, exog] array.

    Attributes
    ----------
    results_final : instance
        Results instance returned by the final fit of the penalized model, i.e.
        after trimming exog with params below trimming threshold.
    results_pen : results instance
        Results instance of the penalized model before trimming. This includes
        variables from the last forward selection
    idx_nonzero
        index of exog columns in the final selection including exog_keep
    idx_exog
        index of exog columns in the final selection for exog candidates, i.e.
        without exog_keep
    idx_excl
        idx of excluded exog based on combined [exog_keep, exog] array. This is
        the complement of idx_nonzero
    converged : bool
        True if the iteration has converged and stopped before maxiter has been
        reached. False if maxiter has been reached.
    iterations : int
        number of iterations in the screening process. Each iteration consists
        of a forward selection step and a trimming step.
    history : dict of lists
        results collected for each iteration during the screening process
        'idx_nonzero' 'params_keep'].append(start_params)
            history['idx_added'].append(idx)

    The ScreeningResults returned by `screen_exog_iterator` has additional
    attributes:

    idx_nonzero_batches : ndarray 2-D
        Two-dimensional array with batch index in the first column and variable
        index withing batch in the second column. They can be used jointly as
        index for the data in the exog_iterator.
    exog_final_names : list[str]
        'var<bidx>_<idx>' where `bidx` is the batch index and `idx` is the
        index of the selected column withing batch `bidx`.
    history_batches : dict of lists
        This provides information about the selected variables within each
        batch during the first round screening
        'idx_nonzero' is based ond the array that includes exog_keep, while
        'idx_exog' is the index based on the exog of the batch.
    """

    def __init__(self, screener, **kwds):
        self.screener = screener
        self.__dict__.update(**kwds)