import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import scipy.special as sc
from scipy.special import rgamma, wright_bessel
Test cases of test_data that do not reach relative accuracy of 1e-11

    Here we test for reduced accuracy or even nan.
    