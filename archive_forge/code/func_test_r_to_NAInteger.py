import pytest
import math
import rpy2.rinterface as ri
def test_r_to_NAInteger():
    na_int = ri.NA_Integer
    r_na_int = ri.evalr('NA_integer_')[0]
    assert r_na_int is na_int