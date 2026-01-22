import collections
import dataclasses
import enum
import itertools as it
import logging
from typing import (
from typing_extensions import Literal
import torch
from torch._C import FunctionSchema
from torch._C._autograd import _ProfilerResult
from torch._C._profiler import (
from torch._utils import _element_size
from torch.profiler import _utils
def _set_activations(self) -> None:
    """Flood the graph to identify activations."""
    required = {Category.INPUT, Category.ACTIVATION}
    also_allowed = {Category.PARAMETER, Category.TEMPORARY}
    for node in self._data_flow_graph.flow_nodes:
        inputs = {(key, value) for key, (_, value) in node.inputs.items()}
        input_categories = {self._categories.get(*i) for i in inputs}
        if input_categories & required and (not input_categories - (required | also_allowed)) and (RecordScope.BACKWARD_FUNCTION not in get_scopes(node._event)):
            for i in node.outputs.items():
                self._categories.setdefault_by_version(*i, Category.ACTIVATION)