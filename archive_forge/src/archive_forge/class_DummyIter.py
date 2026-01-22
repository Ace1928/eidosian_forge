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
class DummyIter(mx.io.DataIter):
    """A dummy iterator that always returns the same batch of data
    (the first data batch of the real data iter). This is usually used for speed testing.

    Parameters
    ----------
    real_iter: mx.io.DataIter
        The real data iterator where the first batch of data comes from
    """

    def __init__(self, real_iter):
        super(DummyIter, self).__init__()
        self.real_iter = real_iter
        self.provide_data = real_iter.provide_data
        self.provide_label = real_iter.provide_label
        self.batch_size = real_iter.batch_size
        self.the_batch = next(real_iter)

    def __iter__(self):
        return self

    def next(self):
        """Get a data batch from iterator. The first data batch of real iter is always returned.
        StopIteration will never be raised.

        Returns
        -------
        DataBatch
            The data of next batch.
        """
        return self.the_batch