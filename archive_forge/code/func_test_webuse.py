import os
from ssl import SSLError
from socket import timeout
from urllib.error import HTTPError, URLError
import numpy as np
from numpy.testing import assert_, assert_array_equal
import pytest
from statsmodels.datasets import get_rdataset, webuse, check_internet, utils
def test_webuse():
    from statsmodels.iolib.tests.results.macrodata import macrodata_result
    res2 = np.array([list(row) for row in macrodata_result])
    base_gh = 'https://github.com/statsmodels/statsmodels/raw/main/statsmodels/datasets/macrodata/'
    internet_available = check_internet(base_gh)
    if not internet_available:
        pytest.skip('Unable to retrieve file - skipping test')
    try:
        res1 = webuse('macrodata', baseurl=base_gh, as_df=False)
    except IGNORED_EXCEPTIONS:
        pytest.skip('Failed with HTTPError or URLError, these are random')
    assert_array_equal(res1, res2)