import collections
import csv
import re
from typing import (Any, Callable, Dict, IO, Iterable, List, Mapping, Optional,
import numpy as np
from tensorflow.lite.python import convert
from tensorflow.lite.python import interpreter as _interpreter
from tensorflow.lite.python.metrics import metrics as metrics_stub  # type: ignore
from tensorflow.python.util import tf_export
def _initialize_stats(self):
    """Helper function initializes stats."""
    self._defining_op = dict()
    for op_info in self._quant_interpreter._get_ops_details():
        self._defining_op.update({tensor_idx: op_info['index'] for tensor_idx in op_info['outputs']})
    self._numeric_verify_tensor_details = None
    self._numeric_verify_op_details = None
    if not self._get_numeric_verify_tensor_details():
        raise ValueError('Please check if the quantized model is in debug mode')
    self._layer_debug_metrics = _DEFAULT_LAYER_DEBUG_METRICS.copy()
    if self._debug_options.layer_debug_metrics:
        self._layer_debug_metrics.update(self._debug_options.layer_debug_metrics)
    self.layer_statistics = None
    self.model_statistics = None
    self._metrics = metrics_stub.TFLiteMetrics()
    self._metrics.increase_counter_debugger_creation()