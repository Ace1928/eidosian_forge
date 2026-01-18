import os
from collections import OrderedDict
from os.path import join as pjoin, dirname
from glob import glob
from io import BytesIO
import re
from tempfile import mkdtemp
import warnings
import shutil
import gzip
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import array
import scipy.sparse as SP
import scipy.io
from scipy.io.matlab import MatlabOpaque, MatlabFunction, MatlabObject
import scipy.io.matlab._byteordercodes as boc
from scipy.io.matlab._miobase import (
from scipy.io.matlab._mio import mat_reader_factory, loadmat, savemat, whosmat
from scipy.io.matlab._mio5 import (
import scipy.io.matlab._mio5_params as mio5p
from scipy._lib._util import VisibleDeprecationWarning
def types_compatible(var1, var2):
    """Check if types are same or compatible.

    0-D numpy scalars are compatible with bare python scalars.
    """
    type1 = type(var1)
    type2 = type(var2)
    if type1 is type2:
        return True
    if type1 is np.ndarray and var1.shape == ():
        return type(var1.item()) is type2
    if type2 is np.ndarray and var2.shape == ():
        return type(var2.item()) is type1
    return False