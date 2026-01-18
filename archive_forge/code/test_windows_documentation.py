import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
This test ensures the correctness of the implemented Taylor
        Windowing function. A Taylor Window of 1024 points is created, its FFT
        is taken, and the Peak Sidelobe Level (PSLL) and 3dB and 18dB bandwidth
        are found and checked.

        A publication from Sandia National Laboratories was used as reference
        for the correctness values [1]_.

        References
        -----
        .. [1] Armin Doerry, "Catalog of Window Taper Functions for
               Sidelobe Control", 2017.
               https://www.researchgate.net/profile/Armin_Doerry/publication/316281181_Catalog_of_Window_Taper_Functions_for_Sidelobe_Control/links/58f92cb2a6fdccb121c9d54d/Catalog-of-Window-Taper-Functions-for-Sidelobe-Control.pdf
        