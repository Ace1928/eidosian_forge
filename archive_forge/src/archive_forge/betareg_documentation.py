import numpy as np
from scipy.special import gammaln as lgamma
import patsy
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import (
from statsmodels.genmod import families

        Get an instance of MLEInfluence with influence and outlier measures

        Returns
        -------
        infl : MLEInfluence instance
            The instance has methods to calculate the main influence and
            outlier measures as attributes.

        See Also
        --------
        statsmodels.stats.outliers_influence.MLEInfluence

        Notes
        -----
        Support for mutli-link and multi-exog models is still experimental
        in MLEInfluence. Interface and some definitions might still change.

        Note: Difference to R betareg: Betareg has the same general leverage
        as this model. However, they use a linear approximation hat matrix
        to scale and studentize influence and residual statistics.
        MLEInfluence uses the generalized leverage as hat_matrix_diag.
        Additionally, MLEInfluence uses pearson residuals for residual
        analusis.

        References
        ----------
        todo

        