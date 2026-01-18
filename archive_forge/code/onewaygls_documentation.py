import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS, WLS

        generic likelihood ratio test between nested models

            \begin{align}
            D & = -2(\ln(\text{likelihood for null model}) - \ln(\text{likelihood for alternative model})) \\
            & = -2\ln\left( \frac{\text{likelihood for null model}}{\text{likelihood for alternative model}} \right).
            \end{align}

        is distributed as chisquare with df equal to difference in number of parameters or equivalently
        difference in residual degrees of freedom  (sign?)

        TODO: put into separate function
        