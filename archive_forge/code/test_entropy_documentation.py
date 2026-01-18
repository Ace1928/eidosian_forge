import numpy as np
from numpy.testing import assert_equal, assert_allclose
from numpy.testing import (assert_, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.stats as stats

    Vasicek results are compared with the R package vsgoftest.

    # library(vsgoftest)
    #
    # samp <- c(<values>)
    # entropy.estimate(x = samp, window = <window_length>)

    