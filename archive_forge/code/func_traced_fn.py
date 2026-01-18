from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401
import torch.nn.functional as F
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
from torch.testing._internal.common_nn import module_tests, new_module_tests
from torch.testing._internal.common_utils import is_iterable_of_tensors
import collections
from copy import deepcopy
from typing import Any, Dict, List, Union
import math  # noqa: F401
from torch import inf
def traced_fn(*inputs, **kwargs):
    fn_tensors, split_inputs = partial_apply_nontensors(fn, inputs, kwargs)
    if not cache_traced_fn or not hasattr(traced_fn, 'traced'):
        traced = torch.jit.trace(fn_tensors, split_inputs.all_tensors, check_trace=False)
        self.assertExportImport(traced.graph, split_inputs.all_tensors)
        output = traced(*split_inputs.all_tensors)
        if cache_traced_fn:
            traced_fn.traced = traced
            traced_fn.split_inputs = split_inputs
    else:
        self.assertTrue(traced_fn.split_inputs.nontensors_match(split_inputs))
        output = traced_fn.traced(*split_inputs.all_tensors)
        traced = traced_fn.traced
    traced_fn.last_graph = traced.graph_for(*split_inputs.all_tensors)
    traced_fn.graph = traced.graph
    return output