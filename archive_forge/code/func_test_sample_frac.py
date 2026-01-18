import pytest
from rpy2.robjects import packages
from rpy2.robjects import rl
from rpy2.robjects.vectors import StrVector
def test_sample_frac(self):
    dataf_a = dplyr.DataFrame(mtcars)
    res = dataf_a.sample_frac(0.5)
    assert res.nrow == int(dataf_a.nrow / 2)