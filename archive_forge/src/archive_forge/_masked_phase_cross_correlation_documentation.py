from functools import partial
import numpy as np
import scipy.fft as fftmodule
from scipy.fft import next_fast_len
from .._shared.utils import _supported_float_type
Reverse array over many axes. Generalization of arr[::-1] for many
    dimensions. If `axes` is `None`, flip along all axes.