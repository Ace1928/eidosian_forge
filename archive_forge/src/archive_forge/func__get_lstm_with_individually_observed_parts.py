import copy
import operator
import torch
from typing import Any, Callable, Optional, Tuple
from torch.ao.quantization import (
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.observer import _PartialWrapper
from torch.ao.quantization.quantize_fx import (
def _get_lstm_with_individually_observed_parts(float_lstm: torch.nn.LSTM, example_inputs: Tuple[Any, ...], backend_config: Optional[BackendConfig]=None, linear_output_obs_ctr: Optional[_PartialWrapper]=None, sigmoid_obs_ctr: Optional[_PartialWrapper]=None, tanh_obs_ctr: Optional[_PartialWrapper]=None, cell_state_obs_ctr: Optional[_PartialWrapper]=None, hidden_state_obs_ctr: Optional[_PartialWrapper]=None) -> torch.ao.nn.quantizable.LSTM:
    """
    Return an observed `torch.ao.nn.quantizable.LSTM` created from a `torch.nn.LSTM`
    with specific observers or fake quantizes assigned to the inner ops or submodules.

    In both eager and FX graph mode quantization, `torch.ao.nn.quantizable.LSTM` is
    used as an observed custom module, which is responsible for inserting its own
    observers. By default, all inner ops inherit the parent custom module's QConfig.
    Users who wish to override this behavior may extend `torch.ao.nn.quantizable.LSTM`
    and use this helper function to customize the observer insertion logic.

    This is meant to be used to convert a float module to an observed module in the
    custom module flow.

    Args:
        `float_lstm`: The float LSTM module
        `example_inputs`: example inputs for the forward function of the LSTM module
        `backend_config`: BackendConfig to use to observe the LSTM module
        `linear_output_obs_ctr`: observer or fake quantize for linear outputs Wx + b,
            where W is the weight matrix, b is the bias, and x is either the inputs
            or the hidden state from the previous layer (if any)
        `sigmoid_obs_ctr`: observer or fake quantize for sigmoid activations
        `tanh_obs_ctr`: observer or fake quantize for tanh activations
        `cell_state_obs_ctr`: observer or fake quantize for the cell state
        `hidden_state_obs_ctr`: observer or fake quantize for the hidden state and
            the output

    Return:
        A `torch.ao.nn.quantizable.LSTM` with the specified observers or fake quantizes
        assigned to the inner ops.
    """

    def make_qconfig(obs_ctr: _PartialWrapper) -> QConfig:
        """
        Make a QConfig with fixed qparams observers or fake quantizes.
        """
        if isinstance(obs_ctr(), FakeQuantizeBase):
            weight = default_weight_fake_quant
        else:
            weight = default_weight_observer
        return QConfig(activation=obs_ctr, weight=weight)
    quantizable_lstm = torch.ao.nn.quantizable.LSTM(float_lstm.input_size, float_lstm.hidden_size, float_lstm.num_layers, float_lstm.bias, float_lstm.batch_first, float_lstm.dropout, float_lstm.bidirectional)
    quantizable_lstm.qconfig = float_lstm.qconfig
    for idx in range(float_lstm.num_layers):
        quantizable_lstm.layers[idx] = torch.ao.nn.quantizable.modules.rnn._LSTMLayer.from_float(float_lstm, idx, float_lstm.qconfig, batch_first=False)
    cell_qm = QConfigMapping().set_global(float_lstm.qconfig)
    if sigmoid_obs_ctr is not None:
        cell_qm.set_module_name('input_gate', make_qconfig(sigmoid_obs_ctr))
        cell_qm.set_module_name('forget_gate', make_qconfig(sigmoid_obs_ctr))
        cell_qm.set_module_name('output_gate', make_qconfig(sigmoid_obs_ctr))
    if tanh_obs_ctr is not None:
        cell_qm.set_module_name('cell_gate', make_qconfig(tanh_obs_ctr))
    for layer in quantizable_lstm.layers:
        cell = layer.layer_fw.cell
        cell = prepare_fx(cell, cell_qm, example_inputs, backend_config=backend_config)
        op_index_to_activation_post_process_ctr = {(torch.add, 0): linear_output_obs_ctr, (torch.mul, 0): cell_state_obs_ctr, (torch.mul, 1): cell_state_obs_ctr, (torch.add, 1): cell_state_obs_ctr, (torch.mul, 2): hidden_state_obs_ctr}
        add_count = 0
        mul_count = 0
        for node in cell.graph.nodes:
            op_index: Optional[Tuple[Callable, int]] = None
            if node.target == torch.add:
                op_index = (torch.add, add_count)
                add_count += 1
            elif node.target == torch.mul:
                op_index = (torch.mul, mul_count)
                mul_count += 1
            else:
                continue
            if op_index not in op_index_to_activation_post_process_ctr:
                continue
            assert len(node.users) == 1
            activation_post_process_name = next(iter(node.users.keys())).name
            activation_post_process_ctr = op_index_to_activation_post_process_ctr[op_index]
            if activation_post_process_ctr is not None:
                setattr(cell, activation_post_process_name, activation_post_process_ctr())
        layer.layer_fw.cell = cell
    return quantizable_lstm