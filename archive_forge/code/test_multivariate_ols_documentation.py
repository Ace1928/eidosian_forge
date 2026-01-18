import numpy as np
import pandas as pd
from statsmodels.multivariate.multivariate_ols import _MultivariateOLS
from numpy.testing import assert_array_almost_equal, assert_raises
import patsy
 Testing within-subject effect interact with 2 between-subject effect
    Compares with R car library Anova(, type=3) output

    Note: The test statistis Phillai, Wilks, Hotelling-Lawley
          and Roy are the same as R output but the approximate F and degree
          of freedoms can be different. This is due to the fact that this
          implementation is based on SAS formula [1]

    .. [*] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm
    