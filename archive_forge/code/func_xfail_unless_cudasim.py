import os
import platform
import shutil
from numba.tests.support import SerialMixin
from numba.cuda.cuda_paths import get_conda_ctk
from numba.cuda.cudadrv import driver, devices, libs
from numba.core import config
from numba.tests.support import TestCase
from pathlib import Path
import unittest
def xfail_unless_cudasim(fn):
    if config.ENABLE_CUDASIM:
        return fn
    else:
        return unittest.expectedFailure(fn)