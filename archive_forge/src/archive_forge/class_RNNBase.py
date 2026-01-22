import numbers
import warnings
import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Tuple, Optional, List, Union, Dict  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
from torch.ao.nn.quantized.modules.utils import _quantize_weight
class RNNBase(torch.nn.Module):
    _FLOAT_MODULE = nn.RNNBase
    _version = 2

    def __init__(self, mode, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, dtype=torch.qint8):
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.dtype = dtype
        self.version = 2
        self.training = False
        num_directions = 2 if bidirectional else 1
        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or isinstance(dropout, bool):
            raise ValueError('dropout should be a number in range [0, 1] representing the probability of an element being zeroed')
        if dropout > 0 and num_layers == 1:
            warnings.warn(f'dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout={dropout} and num_layers={num_layers}')
        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        else:
            raise ValueError('Unrecognized RNN mode: ' + mode)
        _all_weight_values = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                w_ih = torch.randn(gate_size, layer_input_size).to(torch.float)
                w_hh = torch.randn(gate_size, hidden_size).to(torch.float)
                b_ih = torch.randn(gate_size).to(torch.float)
                b_hh = torch.randn(gate_size).to(torch.float)
                if dtype == torch.qint8:
                    w_ih = torch.quantize_per_tensor(w_ih, scale=0.1, zero_point=0, dtype=torch.qint8)
                    w_hh = torch.quantize_per_tensor(w_hh, scale=0.1, zero_point=0, dtype=torch.qint8)
                    packed_ih = torch.ops.quantized.linear_prepack(w_ih, b_ih)
                    packed_hh = torch.ops.quantized.linear_prepack(w_hh, b_hh)
                    if self.version is None or self.version < 2:
                        cell_params = torch.ops.quantized.make_quantized_cell_params_dynamic(packed_ih, packed_hh, b_ih, b_hh)
                    else:
                        cell_params = torch.ops.quantized.make_quantized_cell_params_dynamic(packed_ih, packed_hh, b_ih, b_hh, True)
                else:
                    packed_ih = torch.ops.quantized.linear_prepack_fp16(w_ih, b_ih)
                    packed_hh = torch.ops.quantized.linear_prepack_fp16(w_hh, b_hh)
                    cell_params = torch.ops.quantized.make_quantized_cell_params_fp16(packed_ih, packed_hh)
                _all_weight_values.append(PackedParameter(cell_params))
        self._all_weight_values = torch.nn.ModuleList(_all_weight_values)

    def _get_name(self):
        return 'DynamicQuantizedRNN'

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            if isinstance(module, (PackedParameter, nn.ModuleList)):
                continue
            mod_str = repr(module)
            mod_str = nn.modules.module._addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines
        main_str = self._get_name() + '('
        if lines:
            if len(extra_lines) == 1 and (not child_lines):
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str

    def check_input(self, input: Tensor, batch_sizes: Optional[Tensor]) -> None:
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(f'input must have {expected_input_dim} dimensions, got {input.dim()}')
        if self.input_size != input.size(-1):
            raise RuntimeError(f'input.size(-1) must be equal to input_size. Expected {self.input_size}, got {input.size(-1)}')

    def get_expected_hidden_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions, mini_batch, self.hidden_size)
        return expected_hidden_size

    def check_hidden_size(self, hx: Tensor, expected_hidden_size: Tuple[int, int, int], msg: str='Expected hidden size {}, got {}') -> None:
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, list(hx.size())))

    def check_forward_args(self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]) -> None:
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)
        self.check_hidden_size(hidden, expected_hidden_size, msg='Expected hidden size {}, got {}')

    def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]) -> Tensor:
        if permutation is None:
            return hx
        return _apply_permutation(hx, permutation)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        self.version = version
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs)

    def set_weight_bias(self, weight_bias_dict):

        def weight_bias_name(ihhh, layer, suffix):
            weight_name = f'weight_{ihhh}_l{layer}{suffix}'
            bias_name = f'bias_{ihhh}_l{layer}{suffix}'
            return (weight_name, bias_name)
        num_directions = 2 if self.bidirectional else 1
        _all_weight_values = []
        for layer in range(self.num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                w_ih_name, b_ih_name = weight_bias_name('ih', layer, suffix)
                w_hh_name, b_hh_name = weight_bias_name('hh', layer, suffix)
                w_ih = weight_bias_dict[w_ih_name]
                b_ih = weight_bias_dict[b_ih_name]
                w_hh = weight_bias_dict[w_hh_name]
                b_hh = weight_bias_dict[b_hh_name]
                if w_ih.dtype == torch.qint8:
                    packed_ih = torch.ops.quantized.linear_prepack(w_ih, b_ih)
                    packed_hh = torch.ops.quantized.linear_prepack(w_hh, b_hh)
                    if self.version is None or self.version < 2:
                        cell_params = torch.ops.quantized.make_quantized_cell_params_dynamic(packed_ih, packed_hh, b_ih, b_hh)
                    else:
                        cell_params = torch.ops.quantized.make_quantized_cell_params_dynamic(packed_ih, packed_hh, b_ih, b_hh, True)
                else:
                    packed_ih = torch.ops.quantized.linear_prepack_fp16(w_ih, b_ih)
                    packed_hh = torch.ops.quantized.linear_prepack_fp16(w_hh, b_hh)
                    cell_params = torch.ops.quantized.make_quantized_cell_params_fp16(packed_ih, packed_hh)
                _all_weight_values.append(PackedParameter(cell_params))
        self._all_weight_values = torch.nn.ModuleList(_all_weight_values)

    @classmethod
    def from_float(cls, mod):
        assert type(mod) in {torch.nn.LSTM, torch.nn.GRU}, 'nn.quantized.dynamic.RNNBase.from_float only works for nn.LSTM and nn.GRU'
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        if mod.qconfig is not None and mod.qconfig.weight is not None:
            weight_observer_method = mod.qconfig.weight
        else:
            from torch.ao.quantization.qconfig import default_dynamic_qconfig
            weight_observer_method = default_dynamic_qconfig.weight
        dtype = weight_observer_method().dtype
        supported_scalar_types = [torch.qint8, torch.float16]
        if dtype not in supported_scalar_types:
            raise RuntimeError(f'Unsupported dtype for dynamic RNN quantization: {dtype}')
        qRNNBase: Union[LSTM, GRU]
        if mod.mode == 'LSTM':
            qRNNBase = LSTM(mod.input_size, mod.hidden_size, mod.num_layers, mod.bias, mod.batch_first, mod.dropout, mod.bidirectional, dtype)
        elif mod.mode == 'GRU':
            qRNNBase = GRU(mod.input_size, mod.hidden_size, mod.num_layers, mod.bias, mod.batch_first, mod.dropout, mod.bidirectional, dtype)
        else:
            raise NotImplementedError('Only LSTM/GRU is supported for QuantizedRNN for now')
        num_directions = 2 if mod.bidirectional else 1
        assert mod.bias
        _all_weight_values = []
        for layer in range(qRNNBase.num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''

                def retrieve_weight_bias(ihhh):
                    weight_name = f'weight_{ihhh}_l{layer}{suffix}'
                    bias_name = f'bias_{ihhh}_l{layer}{suffix}'
                    weight = getattr(mod, weight_name)
                    bias = getattr(mod, bias_name)
                    return (weight, bias)
                weight_ih, bias_ih = retrieve_weight_bias('ih')
                weight_hh, bias_hh = retrieve_weight_bias('hh')
                if dtype == torch.qint8:

                    def quantize_and_pack(w, b):
                        weight_observer = weight_observer_method()
                        weight_observer(w)
                        qweight = _quantize_weight(w.float(), weight_observer)
                        packed_weight = torch.ops.quantized.linear_prepack(qweight, b)
                        return packed_weight
                    packed_ih = quantize_and_pack(weight_ih, bias_ih)
                    packed_hh = quantize_and_pack(weight_hh, bias_hh)
                    if qRNNBase.version is None or qRNNBase.version < 2:
                        cell_params = torch.ops.quantized.make_quantized_cell_params_dynamic(packed_ih, packed_hh, bias_ih, bias_hh)
                    else:
                        cell_params = torch.ops.quantized.make_quantized_cell_params_dynamic(packed_ih, packed_hh, bias_ih, bias_hh, True)
                elif dtype == torch.float16:
                    packed_ih = torch.ops.quantized.linear_prepack_fp16(weight_ih.float(), bias_ih)
                    packed_hh = torch.ops.quantized.linear_prepack_fp16(weight_hh.float(), bias_hh)
                    cell_params = torch.ops.quantized.make_quantized_cell_params_fp16(packed_ih, packed_hh)
                else:
                    raise RuntimeError('Unsupported dtype specified for dynamic quantized LSTM!')
                _all_weight_values.append(PackedParameter(cell_params))
        qRNNBase._all_weight_values = torch.nn.ModuleList(_all_weight_values)
        return qRNNBase

    def _weight_bias(self):
        weight_bias_dict: Dict[str, Dict] = {'weight': {}, 'bias': {}}
        count = 0
        num_directions = 2 if self.bidirectional else 1
        for layer in range(self.num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                key_name1 = f'weight_ih_l{layer}{suffix}'
                key_name2 = f'weight_hh_l{layer}{suffix}'
                packed_weight_bias = self._all_weight_values[count].param.__getstate__()[0][4]
                weight_bias_dict['weight'][key_name1] = packed_weight_bias[0].__getstate__()[0][0]
                weight_bias_dict['weight'][key_name2] = packed_weight_bias[1].__getstate__()[0][0]
                key_name1 = f'bias_ih_l{layer}{suffix}'
                key_name2 = f'bias_hh_l{layer}{suffix}'
                weight_bias_dict['bias'][key_name1] = packed_weight_bias[0].__getstate__()[0][1]
                weight_bias_dict['bias'][key_name2] = packed_weight_bias[1].__getstate__()[0][1]
                count = count + 1
        return weight_bias_dict

    def get_weight(self):
        return self._weight_bias()['weight']

    def get_bias(self):
        return self._weight_bias()['bias']