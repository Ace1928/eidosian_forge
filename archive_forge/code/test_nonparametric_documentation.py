from statsmodels.compat.python import lzip
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
from scipy import stats
import pytest
from statsmodels.stats.contingency_tables import (
from statsmodels.sandbox.stats.runs import (Runs,
from statsmodels.sandbox.stats.runs import mcnemar as sbmcnemar
from statsmodels.stats.nonparametric import (
from statsmodels.tools.testing import Holder
expand a 2 by 2 contingency table to observations
    