from two mean values what can be explained by the data and
from textwrap import dedent
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
def two_fold(self, std=False, two_fold_type='pooled', submitted_weight=None, n=None, conf=None):
    """
        Calculates the two-fold or pooled Oaxaca Blinder Decompositions

        Methods
        -------
        std: boolean, optional
            If true, bootstrapped standard errors will be calculated.

        two_fold_type: string, optional
            This method allows for the specific calculation of the
            non-discriminatory model. There are four different types
            available at this time. pooled, cotton, reimers, self_submitted.
            Pooled is assumed and if a non-viable parameter is given,
            pooled will be ran.

            pooled - This type assumes that the pooled model's parameters
            (a normal regression) is the non-discriminatory model.
            This includes the indicator variable. This is generally
            the best idea. If you have economic justification for
            using others, then use others.

            nuemark - This is similar to the pooled type, but the regression
            is not done including the indicator variable.

            cotton - This type uses the adjusted in Cotton (1988), which
            accounts for the undervaluation of one group causing the
            overevalution of another. It uses the sample size weights for
            a linear combination of the two model parameters

            reimers - This type uses a linear combination of the two
            models with both parameters being 50% of the
            non-discriminatory model.

            self_submitted - This allows the user to submit their
            own weights. Please be sure to put the weight of the larger mean
            group only. This should be submitted in the
            submitted_weights variable.

        submitted_weight: int/float, required only for self_submitted,
            This is the submitted weight for the larger mean. If the
            weight for the larger mean is p, then the weight for the
            other mean is 1-p. Only submit the first value.

        n: int, optional
            A amount of iterations to calculate the bootstrapped
            standard errors. This defaults to 5000.
        conf: float, optional
            This is the confidence required for the standard error
            calculation. Defaults to .99, but could be anything less
            than or equal to one. One is heavy discouraged, due to the
            extreme outliers inflating the variance.

        Returns
        -------
        OaxacaResults
            A results container for the two-fold decomposition.
        """
    self.submitted_n = n
    self.submitted_conf = conf
    std_val = None
    self.two_fold_type = two_fold_type
    self.submitted_weight = submitted_weight
    if two_fold_type == 'cotton':
        self.t_params = self.len_f / (self.len_f + self.len_s) * self._f_model.params + self.len_s / (self.len_f + self.len_s) * self._s_model.params
    elif two_fold_type == 'reimers':
        self.t_params = 0.5 * (self._f_model.params + self._s_model.params)
    elif two_fold_type == 'self_submitted':
        if submitted_weight is None:
            raise ValueError('Please submit weights')
        submitted_weight = [submitted_weight, 1 - submitted_weight]
        self.t_params = submitted_weight[0] * self._f_model.params + submitted_weight[1] * self._s_model.params
    elif two_fold_type == 'nuemark':
        self._t_model = OLS(self.endog, self.neumark).fit(cov_type=self.cov_type, cov_kwds=self.cov_kwds)
        self.t_params = self._t_model.params
    else:
        self._t_model = OLS(self.endog, self.exog).fit(cov_type=self.cov_type, cov_kwds=self.cov_kwds)
        self.t_params = np.delete(self._t_model.params, self.bifurcate)
    self.unexplained = self.exog_f_mean @ (self._f_model.params - self.t_params) + self.exog_s_mean @ (self.t_params - self._s_model.params)
    self.explained = (self.exog_f_mean - self.exog_s_mean) @ self.t_params
    if std is True:
        std_val = self.variance(2)
    return OaxacaResults((self.unexplained, self.explained, self.gap), 2, std_val=std_val)