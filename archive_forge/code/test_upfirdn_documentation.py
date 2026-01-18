import numpy as np
from itertools import product
from numpy.testing import assert_equal, assert_allclose
from pytest import raises as assert_raises
import pytest
from scipy.signal import upfirdn, firwin
from scipy.signal._upfirdn import _output_len, _upfirdn_modes
from scipy.signal._upfirdn_apply import _pad_test
Test vs. manually computed results for modes not in numpy's pad.