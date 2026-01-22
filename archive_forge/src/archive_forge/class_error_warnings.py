from __future__ import annotations
import os
import re
import sys
import typing as ty
import unittest
import warnings
from contextlib import nullcontext
from itertools import zip_longest
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .helpers import assert_data_similar, bytesio_filemap, bytesio_round_trip
from .np_features import memmap_after_ufunc
class error_warnings(clear_and_catch_warnings):
    """Context manager to check for warnings as errors.  Usually used with
    ``assert_raises`` in the with block

    Examples
    --------
    >>> with error_warnings():
    ...     try:
    ...         warnings.warn('Message', UserWarning)
    ...     except UserWarning:
    ...         print('I consider myself warned')
    I consider myself warned
    """
    filter = 'error'

    def __enter__(self):
        mgr = super().__enter__()
        warnings.simplefilter(self.filter)
        return mgr