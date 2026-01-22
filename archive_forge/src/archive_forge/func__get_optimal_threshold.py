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
def _get_optimal_threshold(hist_data, quantized_dtype, num_quantized_bins=255):
    """Given a dataset, find the optimal threshold for quantizing it.
    The reference distribution is `q`, and the candidate distribution is `p`.
    `q` is a truncated version of the original distribution.

    Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
    """
    hist, hist_edges, min_val, max_val, _ = hist_data
    num_bins = len(hist)
    assert num_bins % 2 == 1
    if min_val >= 0 and quantized_dtype in ['auto', 'uint8']:
        num_quantized_bins = num_quantized_bins * 2 + 1
    hist = ndarray.array(hist, ctx=cpu())
    hist_edges = ndarray.array(hist_edges, ctx=cpu())
    threshold, divergence = ndarray.contrib.calibrate_entropy(hist=hist, hist_edges=hist_edges, num_quantized_bins=num_quantized_bins)
    threshold = threshold.asnumpy()
    divergence = divergence.asnumpy()
    return (min_val, max_val, threshold, divergence)