import ctypes
import logging
import os
import shutil
import warnings
import numpy as np
from ..base import _LIB, check_call, py_str
from ..base import c_array, c_str, mx_uint, c_str_array
from ..base import NDArrayHandle, SymbolHandle
from ..symbol import Symbol
from ..symbol import load as sym_load
from .. import ndarray
from ..ndarray import load as nd_load
from ..ndarray import save as nd_save
from ..ndarray import NDArray
from ..io import DataIter, DataDesc, DataBatch
from ..context import cpu, Context
from ..module import Module
class _LayerHistogramCollector(object):
    """Saves layer histogram in a dict with layer names as keys and lists of NDArrays as
    values. The collected histogram will be used for calculating the optimal thresholds for
    quantization using KL divergence.
    """

    def __init__(self, num_bins=8001, include_layer=None, logger=None):
        self.hist_dict = {}
        self.num_bins = num_bins
        self.include_layer = include_layer
        self.logger = logger

    def collect(self, name, arr):
        """Callback function for collecting layer output NDArrays."""
        name = py_str(name)
        if name not in self.include_layer:
            return
        handle = ctypes.cast(arr, NDArrayHandle)
        arr = NDArray(handle, writable=False).copyto(cpu()).asnumpy()
        if self.logger:
            self.logger.debug('Collecting layer %s histogram of shape %s' % (name, arr.shape))
        min_range = np.min(arr)
        max_range = np.max(arr)
        th = max(abs(min_range), abs(max_range))
        if name in self.hist_dict:
            self.hist_dict[name] = combine_histogram(self.hist_dict[name], arr, min_range, max_range, th)
        else:
            hist, hist_edges = np.histogram(arr, bins=self.num_bins, range=(-th, th))
            self.hist_dict[name] = (hist, hist_edges, min_range, max_range, th)