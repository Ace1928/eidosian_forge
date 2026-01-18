import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.signal._savitzky_golay import _polyder
Tests that the window_length check is using the correct axis.