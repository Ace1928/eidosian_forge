import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import statsmodels.sandbox.infotheo as infotheo
mutual information of two random variables estimated with kde



    Notes
    -----
    bins='auto' selects the number of bins so that approximately 5 observations
    are expected to be in each bin under the assumption of independence. This
    follows roughly the description in Kahn et al. 2007

    