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
def _collect_layer_output_min_max(mod, data, quantized_dtype, include_layer=None, max_num_examples=None, logger=None):
    """Collect min and max values from layer outputs and save them in
    a dictionary mapped by layer names.
    """
    collector = _LayerOutputMinMaxCollector(quantized_dtype=quantized_dtype, include_layer=include_layer, logger=logger)
    num_examples = _collect_layer_statistics(mod, data, collector, max_num_examples, logger)
    return (collector.min_max_dict, num_examples)