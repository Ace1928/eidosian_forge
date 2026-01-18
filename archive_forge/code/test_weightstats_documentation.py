import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
tests for weightstats, compares with replication

no failures but needs cleanup
update 2012-09-09:
   added test after fixing bug in covariance
   TODOs:
     - I do not remember what all the commented out code is doing
     - should be refactored to use generator or inherited tests
     - still gaps in test coverage
       - value/diff in ttest_ind is tested in test_tost.py
     - what about pandas data structures?

Author: Josef Perktold
License: BSD (3-clause)

