import os
import re
import sys
import ctypes
import textwrap
from typing import final
import warnings
from ctypes.util import find_library
from abc import ABC, abstractmethod
from functools import lru_cache
from contextlib import ContextDecorator
def _get_params_for_sequential_blas_under_openmp(self):
    """Return appropriate params to use for a sequential BLAS call in an OpenMP loop

        This function takes into account the unexpected behavior of OpenBLAS with the
        OpenMP threading layer.
        """
    if self.select(internal_api='openblas', threading_layer='openmp').lib_controllers:
        return {'limits': None, 'user_api': None}
    return {'limits': 1, 'user_api': 'blas'}