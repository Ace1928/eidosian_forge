from collections.abc import Iterable
import copy  # check if needed when dropping python 2.7
import numpy as np
from scipy import optimize
import pandas as pd
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import Logit
from statsmodels.genmod.generalized_linear_model import (
import statsmodels.regression.linear_model as lm
from statsmodels.tools.sm_exceptions import (PerfectSeparationError,
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.linalg import matrix_sqrt
from statsmodels.base._penalized import PenalizedMixin
from statsmodels.gam.gam_penalties import MultivariateGamPenalty
from statsmodels.gam.gam_cross_validation.gam_cross_validation import (
from statsmodels.gam.gam_cross_validation.cross_validators import KFold
def _tranform_predict_exog(self, exog=None, exog_smooth=None, transform=True):
    """Transform original explanatory variables for prediction

        Parameters
        ----------
        exog : array_like, optional
            The values for the linear explanatory variables.
        exog_smooth : array_like
            values for the variables in the smooth terms
        transform : bool, optional
            If transform is False, then ``exog`` is returned unchanged and
            ``x`` is ignored. It is assumed that exog contains the full
            design matrix for the predict observations.
            If transform is True, then the basis representation of the smooth
            term will be constructed from the provided ``x``.

        Returns
        -------
        exog_transformed : ndarray
            design matrix for the prediction
        """
    if exog_smooth is not None:
        exog_smooth = np.asarray(exog_smooth)
    exog_index = None
    if transform is False:
        if exog_smooth is None:
            ex = exog
        elif exog is None:
            ex = exog_smooth
        else:
            ex = np.column_stack((exog, exog_smooth))
    else:
        if exog is not None and hasattr(self.model, 'design_info_linear'):
            exog, exog_index = _transform_predict_exog(self.model, exog, self.model.design_info_linear)
        if exog_smooth is not None:
            ex_smooth = self.model.smoother.transform(exog_smooth)
            if exog is None:
                ex = ex_smooth
            else:
                ex = np.column_stack((exog, ex_smooth))
        else:
            ex = exog
    return (ex, exog_index)