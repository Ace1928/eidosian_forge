import collections
import csv
import re
from typing import (Any, Callable, Dict, IO, Iterable, List, Mapping, Optional,
import numpy as np
from tensorflow.lite.python import convert
from tensorflow.lite.python import interpreter as _interpreter
from tensorflow.lite.python.metrics import metrics as metrics_stub  # type: ignore
from tensorflow.python.util import tf_export
def _collect_layer_statistics(self) -> Dict[str, Dict[str, float]]:
    """Collects layer statistics by applying layer debug metrics.

    For all data from the given RepresentativeDataset, collect statistics per
    example by getting the NumericVerify op results in _quant_interpreter
    and calculating layer debug metrics on the results.

    Returns:
      aggregated per-layer statistics of NumericVerify results.
      {layer_name: {metric_name: metric}}
    """
    layer_statistics = collections.defaultdict(lambda: collections.defaultdict(list))
    initialize = True
    for tensor_data in self._data_gen():
        self._set_input_tensors(self._quant_interpreter, tensor_data, initialize)
        initialize = False
        self._quant_interpreter.invoke()
        for tensor_detail in self._get_numeric_verify_tensor_details():
            tensor_name = tensor_detail['name']
            diffs = self._quant_interpreter.get_tensor(tensor_detail['index'])
            for metric_name, metric_fn in self._layer_debug_metrics.items():
                layer_statistics[tensor_name][metric_name].append(metric_fn(diffs))
        if self._debug_options.layer_direct_compare_metrics is not None:
            for tensor_detail in self._get_numeric_verify_tensor_details():
                tensor_name = tensor_detail['name']
                op_idx = self._defining_op[tensor_detail['index']]
                op_detail = self._quant_interpreter._get_op_details(op_idx)
                q_idx, f_idx = op_detail['inputs']
                quant_input_detail = self._quant_interpreter._get_tensor_details(q_idx, subgraph_index=0)
                for metric_name, metric_fn in self._debug_options.layer_direct_compare_metrics.items():
                    layer_statistics[tensor_name][metric_name].append(metric_fn(self._quant_interpreter.get_tensor(f_idx), self._quant_interpreter.get_tensor(q_idx), quant_input_detail['quantization_parameters']['scales'][0], quant_input_detail['quantization_parameters']['zero_points'][0]))
    for metrics in layer_statistics.values():
        for metric_name in metrics:
            metrics[metric_name] = np.nanmean(metrics[metric_name])
    return layer_statistics