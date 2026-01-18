from numbers import Number
import operator
import os
import threading
import contextlib
import numpy as np
from .pypocketfft import good_size
Returns the default number of workers within the current context

    Examples
    --------
    >>> from scipy import fft
    >>> fft.get_workers()
    1
    >>> with fft.set_workers(4):
    ...     fft.get_workers()
    4
    