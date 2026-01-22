import warnings
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lzip
from collections import defaultdict
import numpy as np
from statsmodels.graphics._regressionplots_doc import _plot_influence_doc
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import maybe_unwrap_results
@property
def _fittedvalues_one(self):
    """experimental code
        """
    warnings.warn('this ignores offset and exposure', UserWarning)
    exog = self.results.model.exog
    fitted = np.array([self.results.model.predict(pi, exog[i]) for i, pi in enumerate(self.params_one)])
    return fitted.squeeze()