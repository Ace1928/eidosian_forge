import copy
import operator
import torch
from typing import Any, Callable, Optional, Tuple
from torch.ao.quantization import (
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.observer import _PartialWrapper
from torch.ao.quantization.quantize_fx import (
def _get_reference_quantized_lstm_module(observed_lstm: torch.ao.nn.quantizable.LSTM, backend_config: Optional[BackendConfig]=None) -> torch.ao.nn.quantized.LSTM:
    """
    Return a `torch.ao.nn.quantized.LSTM` created from a `torch.ao.nn.quantizable.LSTM`
    with observers or fake quantizes inserted through `prepare_fx`, e.g. from
    `_get_lstm_with_individually_observed_parts`.

    This is meant to be used to convert an observed module to a quantized module in the
    custom module flow.

    Args:
        `observed_lstm`: a `torch.ao.nn.quantizable.LSTM` observed through `prepare_fx`
        `backend_config`: BackendConfig to use to produce the reference quantized model

    Return:
        A reference `torch.ao.nn.quantized.LSTM` module.
    """
    quantized_lstm = torch.ao.nn.quantized.LSTM(observed_lstm.input_size, observed_lstm.hidden_size, observed_lstm.num_layers, observed_lstm.bias, observed_lstm.batch_first, observed_lstm.dropout, observed_lstm.bidirectional)
    for i, layer in enumerate(quantized_lstm.layers):
        cell = copy.deepcopy(observed_lstm.layers.get_submodule(str(i)).layer_fw.cell)
        cell = convert_to_reference_fx(cell, backend_config=backend_config)
        assert isinstance(cell, torch.fx.GraphModule)
        for node in cell.graph.nodes:
            if node.target == torch.quantize_per_tensor:
                arg = node.args[0]
                if arg.target == 'x' or (arg.target == operator.getitem and arg.args[0].target == 'hidden'):
                    with cell.graph.inserting_before(node):
                        node.replace_all_uses_with(arg)
                        cell.graph.erase_node(node)
            if node.target == 'output':
                for arg in node.args[0]:
                    with cell.graph.inserting_before(node):
                        node.replace_input_with(arg, arg.args[0])
        cell.graph.eliminate_dead_code()
        cell.recompile()
        layer.layer_fw.cell = cell
    return quantized_lstm