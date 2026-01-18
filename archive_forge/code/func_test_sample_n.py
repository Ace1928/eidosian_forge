import pytest
from rpy2.robjects import packages
from rpy2.robjects import rl
from rpy2.robjects.vectors import StrVector
def test_sample_n(self):
    dataf_a = dplyr.DataFrame(mtcars)
    res = dataf_a.sample_n(5)
    assert res.nrow == 5