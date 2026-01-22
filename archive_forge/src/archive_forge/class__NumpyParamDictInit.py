import time
import gzip
import struct
import traceback
import numbers
import sys
import os
import platform
import errno
import logging
import bz2
import zipfile
import json
from contextlib import contextmanager
from collections import OrderedDict
import numpy as np
import numpy.testing as npt
import numpy.random as rnd
import mxnet as mx
from .context import Context, current_context
from .ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID
from .ndarray import array
from .symbol import Symbol
from .symbol.numpy import _Symbol as np_symbol
from .util import use_np, getenv, setenv  # pylint: disable=unused-import
from .runtime import Features
from .numpy_extension import get_cuda_compute_capability
class _NumpyParamDictInit(mx.init.Initializer):
    """Initializes parameters with the cached numpy ndarrays dictionary
        """

    def __init__(self, np_params):
        super(_NumpyParamDictInit, self).__init__()
        self._np_params = np_params

    def _init_weight(self, name, arr):
        arr[()] = self._np_params[name]