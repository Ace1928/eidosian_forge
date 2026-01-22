from __future__ import annotations
import builtins
import functools
import math
import sys
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
import torch._C._onnx as _C_onnx
import torch.nn.modules.utils
import torch.onnx
from torch import _C
from torch.onnx import _constants, _deprecation, _type_utils, errors, symbolic_helper
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
from torch.types import Number
@_beartype.beartype
def _generic_rnn(g: jit_utils.GraphContext, variant, input, initial_states, all_weights, has_biases, num_layers, dropout, train, bidirectional, batch_first=None, batch_sizes=None):
    warnings.warn('Exporting a model to ONNX with a batch_size other than 1, ' + 'with a variable length with ' + variant + ' can cause an error ' + 'when running the ONNX model with a different batch size. ' + 'Make sure to save the model with a batch size of 1, ' + 'or define the initial states (h0/c0) as inputs of the model. ')
    onnxActivations = ['Relu', 'Tanh', 'Sigmoid', 'Affine', 'LeakyRelu', 'ThresholdedRelu', 'ScaledTanh', 'HardSigmoid', 'Elu', 'Softsign', 'Softplus']
    variantToOnnxActivationMap = dict(zip([act_fun.lower() for act_fun in onnxActivations], onnxActivations))
    weights_per_layer = 4 if has_biases else 2
    if variant == 'LSTM' and len(all_weights) != num_layers * weights_per_layer * (1 + bidirectional):
        return symbolic_helper._unimplemented('LSTM', 'LSTMs with projections', input)
    assert len(all_weights) == num_layers * weights_per_layer * (1 + bidirectional)
    layer_weights = [all_weights[i:i + weights_per_layer] for i in range(0, len(all_weights), weights_per_layer)]
    if batch_first:
        input = g.op('Transpose', input, perm_i=[1, 0, 2])
    if dropout and train:
        return symbolic_helper._unimplemented('RNN/GRU/LSTM', 'dropout in training mode', input)
    if variant.startswith('RNN'):
        nonlinearity = variantToOnnxActivationMap[variant[4:].lower()]
        variant = 'RNN'
    w_hh = all_weights[1]
    hidden_size = symbolic_helper._get_tensor_dim_size(w_hh, 1)
    if hidden_size is None:
        return symbolic_helper._unimplemented('RNN/GRU/LSTM', 'unknown hidden size', input)
    unidirectional = not bidirectional
    prev_output = input
    h_outs = []
    if variant == 'RNN' or variant == 'GRU':
        h0 = initial_states
    elif variant == 'LSTM':
        h0, c0 = initial_states
        c_outs = []
    sequence_lens = unused(g) if batch_sizes is None else batch_sizes
    if variant == 'GRU':
        reform_permutation = [(1, 2), (0, 1), (2, 3)]
    elif variant == 'LSTM':
        reform_permutation = [(0, 1), (3, 4), (1, 3)]

    @_beartype.beartype
    def reform_weights(g, w, n, intervals):
        slices = [symbolic_helper._slice_helper(g, w, axes=[0], starts=[x * n], ends=[y * n]) for x, y in intervals]
        return g.op('Concat', *slices, axis_i=0)

    @_beartype.beartype
    def transform_weights_no_bias(layer_index):
        weights = layer_weights[layer_index]
        if variant == 'RNN':
            weight_ih, weight_hh = weights
        elif variant == 'GRU' or variant == 'LSTM':
            weight_ih, weight_hh = (reform_weights(g, w, hidden_size, reform_permutation) for w in weights)
        return tuple((symbolic_helper._unsqueeze_helper(g, x, [0]) for x in (weight_ih, weight_hh)))

    @_beartype.beartype
    def transform_weights(layer_index):
        weights = layer_weights[layer_index]
        if variant == 'RNN':
            weight_ih, weight_hh, bias_ih, bias_hh = weights
        elif variant == 'GRU' or variant == 'LSTM':
            weight_ih, weight_hh, bias_ih, bias_hh = (reform_weights(g, w, hidden_size, reform_permutation) for w in weights)
        bias_concat = g.op('Concat', bias_ih, bias_hh, axis_i=0)
        return tuple((symbolic_helper._unsqueeze_helper(g, x, [0]) for x in (weight_ih, weight_hh, bias_concat)))

    @_beartype.beartype
    def retrieve_state(x, start, end):
        return x if num_layers == 1 else symbolic_helper._slice_helper(g, x, axes=[0], starts=[start], ends=[end])
    for i in range(num_layers):
        if unidirectional:
            if weights_per_layer == 4:
                weight_ih, weight_hh, bias_concat = transform_weights(i)
            else:
                weight_ih, weight_hh = transform_weights_no_bias(i)
                bias_concat = unused(g)
            state_indices = (i, i + 1)
        else:
            if weights_per_layer == 4:
                weight_ih_f, weight_hh_f, bias_f = transform_weights(2 * i)
                weight_ih_b, weight_hh_b, bias_b = transform_weights(2 * i + 1)
                bias_concat = g.op('Concat', bias_f, bias_b, axis_i=0)
            else:
                weight_ih_f, weight_hh_f = transform_weights_no_bias(2 * i)
                weight_ih_b, weight_hh_b = transform_weights_no_bias(2 * i + 1)
                bias_concat = unused(g)
            weight_ih = g.op('Concat', weight_ih_f, weight_ih_b, axis_i=0)
            weight_hh = g.op('Concat', weight_hh_f, weight_hh_b, axis_i=0)
            state_indices = (2 * i, 2 * i + 2)
        inputs = [prev_output, weight_ih, weight_hh, bias_concat, sequence_lens]
        inputs.append(retrieve_state(h0, *state_indices))
        if variant == 'LSTM':
            inputs.append(retrieve_state(c0, *state_indices))
        extra_kwargs = {} if unidirectional else {'direction_s': 'bidirectional'}
        if variant == 'RNN':
            if bidirectional:
                activation = [nonlinearity, nonlinearity]
            else:
                activation = [nonlinearity]
            prev_output, h_out = g.op('RNN', *inputs, outputs=2, hidden_size_i=hidden_size, activations_s=activation, **extra_kwargs)
        elif variant == 'GRU':
            prev_output, h_out = g.op('GRU', *inputs, outputs=2, hidden_size_i=hidden_size, linear_before_reset_i=1, **extra_kwargs)
        elif variant == 'LSTM':
            prev_output, h_out, c_out = g.op('LSTM', *inputs, outputs=3, hidden_size_i=hidden_size, **extra_kwargs)
        if bidirectional:
            prev_output = g.op('Transpose', prev_output, perm_i=[0, 2, 1, 3])
            prev_output = symbolic_helper._reshape_helper(g, prev_output, g.op('Constant', value_t=torch.LongTensor([0, 0, -1])), allowzero=0)
        else:
            prev_output = symbolic_helper._squeeze_helper(g, prev_output, [1])
        h_outs.append(h_out)
        if variant == 'LSTM':
            c_outs.append(c_out)
    if batch_first:
        prev_output = g.op('Transpose', prev_output, perm_i=[1, 0, 2])
    h_outs = h_out if num_layers == 1 else g.op('Concat', *h_outs, axis_i=0)
    if variant == 'RNN' or variant == 'GRU':
        return (prev_output, h_outs)
    elif variant == 'LSTM':
        c_outs = c_out if num_layers == 1 else g.op('Concat', *c_outs, axis_i=0)
        return (prev_output, h_outs, c_outs)