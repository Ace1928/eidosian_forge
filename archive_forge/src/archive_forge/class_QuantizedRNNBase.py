import warnings
from typing import List, Optional, Tuple
import torch
from torch import _VF, Tensor  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
class QuantizedRNNBase(torch.jit.ScriptModule):
    __constants__ = ['mode', 'input_size', 'hidden_size', 'num_layers', 'bias', 'batch_first', 'dropout', 'bidirectional', 'dtype']

    def __init__(self, other, dtype=torch.int8):
        super().__init__()
        warnings.warn('torch.jit.QuantizedRNNBase is deprecated and will be removed in an upcoming PyTorch release. Please use the torch.ao.nn.quantized.dynamic instead.')
        self.mode = other.mode
        self.input_size = other.input_size
        self.hidden_size = other.hidden_size
        self.num_layers = other.num_layers
        self.bias = other.bias
        self.batch_first = other.batch_first
        if self.mode != 'GRU':
            assert not self.batch_first
        self.dropout = other.dropout
        self.bidirectional = other.bidirectional
        num_directions = 2 if self.bidirectional else 1
        self.dtype = dtype
        assert self.bias
        if self.mode != 'LSTM' and self.mode != 'GRU':
            raise RuntimeError('Only LSTM or GRU is supported for QuantizedRNN')
        if dtype != torch.int8 and dtype != torch.float16:
            raise RuntimeError(f'Unsupported dtype: {dtype}')
        self.all_weights = []
        for layer in range(self.num_layers):
            for direction in range(num_directions):
                layer_input_size = self.input_size if layer == 0 else self.hidden_size * num_directions
                suffix = '_reverse' if direction == 1 else ''

                def get_weight_bias(ihhh):
                    weight_name = f'weight_{ihhh}_l{layer}{suffix}'
                    bias_name = f'bias_{ihhh}_l{layer}{suffix}'
                    weight = getattr(other, weight_name)
                    bias = getattr(other, bias_name)
                    return (weight, bias)
                weight_ih, bias_ih = get_weight_bias('ih')
                weight_hh, bias_hh = get_weight_bias('hh')
                if dtype == torch.int8:
                    cell_params = torch.ops.quantized.make_quantized_cell_params(weight_ih, weight_hh, bias_ih, bias_hh)
                else:
                    packed_ih = torch.ops.quantized.linear_prepack_fp16(weight_ih.float(), bias_ih)
                    packed_hh = torch.ops.quantized.linear_prepack_fp16(weight_hh.float(), bias_hh)
                    cell_params = torch.ops.quantized.make_quantized_cell_params_fp16(packed_ih, packed_hh)
                setattr(self, f'cell_params_{layer}_{suffix}', cell_params)
                self.all_weights.append(cell_params)

    @torch.jit.script_method
    def check_input(self, input: Tensor, batch_sizes: Optional[Tensor]) -> None:
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(f'input must have {expected_input_dim} dimensions, got {input.dim()}')
        if self.input_size != input.size(-1):
            raise RuntimeError(f'input.size(-1) must be equal to input_size. Expected {self.input_size}, got {input.size(-1)}')

    @torch.jit.script_method
    def get_expected_hidden_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions, mini_batch, self.hidden_size)
        return expected_hidden_size

    @torch.jit.script_method
    def check_hidden_size(self, hx: Tensor, expected_hidden_size: Tuple[int, int, int], msg: str='Expected hidden size {}, got {}') -> None:
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, list(hx.size())))

    @torch.jit.script_method
    def check_forward_args(self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]) -> None:
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)
        self.check_hidden_size(hidden, expected_hidden_size, msg='Expected hidden size {}, got {}')

    @torch.jit.script_method
    def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]) -> Tensor:
        if permutation is None:
            return hx
        return apply_permutation(hx, permutation)