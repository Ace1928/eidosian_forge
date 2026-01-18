import sys
import pytest
import numpy as np
from typing import NamedTuple
from numpy.testing import assert_allclose
from scipy.special import hyp2f1
from scipy.special._testutils import check_version, MissingModule
@pytest.mark.slow
@check_version(mpmath, '1.0.0')
def test_test_hyp2f1(self):
    """Test that expected values match what is computed by mpmath.

        This gathers the parameters for the test cases out of the pytest marks.
        The parameters are a, b, c, z, expected, rtol, where expected should
        be the value of hyp2f1(a, b, c, z) computed with mpmath. The test
        recomputes hyp2f1(a, b, c, z) using mpmath and verifies that expected
        actually is the correct value. This allows the data for the tests to
        live within the test code instead of an external datafile, while
        avoiding having to compute the results with mpmath during the test,
        except for when slow tests are being run.
        """
    test_methods = [test_method for test_method in dir(self) if test_method.startswith('test') and callable(getattr(self, test_method)) and (test_method != 'test_test_hyp2f1')]
    for test_method in test_methods:
        params = self._get_test_parameters(getattr(self, test_method))
        for a, b, c, z, expected, _ in params:
            assert_allclose(mp_hyp2f1(a, b, c, z), expected, rtol=2.25e-16)