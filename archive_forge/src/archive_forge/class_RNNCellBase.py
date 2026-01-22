import numbers
import warnings
import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Tuple, Optional, List, Union, Dict  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
from torch.ao.nn.quantized.modules.utils import _quantize_weight
class RNNCellBase(torch.nn.Module):
    __constants__ = ['input_size', 'hidden_size', 'bias']

    def __init__(self, input_size, hidden_size, bias=True, num_chunks=4, dtype=torch.qint8):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_dtype = dtype
        if bias:
            self.bias_ih = torch.randn(num_chunks * hidden_size).to(dtype=torch.float)
            self.bias_hh = torch.randn(num_chunks * hidden_size).to(dtype=torch.float)
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        weight_ih = torch.randn(num_chunks * hidden_size, input_size).to(torch.float)
        weight_hh = torch.randn(num_chunks * hidden_size, hidden_size).to(torch.float)
        if dtype == torch.qint8:
            weight_ih = torch.quantize_per_tensor(weight_ih, scale=1, zero_point=0, dtype=torch.qint8)
            weight_hh = torch.quantize_per_tensor(weight_hh, scale=1, zero_point=0, dtype=torch.qint8)
        if dtype == torch.qint8:
            packed_weight_ih = torch.ops.quantized.linear_prepack(weight_ih, self.bias_ih)
            packed_weight_hh = torch.ops.quantized.linear_prepack(weight_hh, self.bias_hh)
        else:
            packed_weight_ih = torch.ops.quantized.linear_prepack_fp16(weight_ih, self.bias_ih)
            packed_weight_hh = torch.ops.quantized.linear_prepack_fp16(weight_hh, self.bias_hh)
        self._packed_weight_ih = packed_weight_ih
        self._packed_weight_hh = packed_weight_hh

    def _get_name(self):
        return 'DynamicQuantizedRNNBase'

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != 'tanh':
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(f'input has inconsistent input_size: got {input.size(1)}, expected {self.input_size}')

    def check_forward_hidden(self, input: Tensor, hx: Tensor, hidden_label: str='') -> None:
        if input.size(0) != hx.size(0):
            raise RuntimeError(f"Input batch size {input.size(0)} doesn't match hidden{hidden_label} batch size {hx.size(0)}")
        if hx.size(1) != self.hidden_size:
            raise RuntimeError(f'hidden{hidden_label} has inconsistent hidden_size: got {hx.size(1)}, expected {self.hidden_size}')

    @classmethod
    def from_float(cls, mod):
        assert type(mod) in {torch.nn.LSTMCell, torch.nn.GRUCell, torch.nn.RNNCell}, 'nn.quantized.dynamic.RNNCellBase.from_float                                  only works for nn.LSTMCell, nn.GRUCell and nn.RNNCell'
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
        qRNNCellBase: Union[LSTMCell, GRUCell, RNNCell]
        if type(mod) == torch.nn.LSTMCell:
            qRNNCellBase = LSTMCell(mod.input_size, mod.hidden_size, bias=mod.bias, dtype=dtype)
        elif type(mod) == torch.nn.GRUCell:
            qRNNCellBase = GRUCell(mod.input_size, mod.hidden_size, bias=mod.bias, dtype=dtype)
        elif type(mod) == torch.nn.RNNCell:
            qRNNCellBase = RNNCell(mod.input_size, mod.hidden_size, bias=mod.bias, nonlinearity=mod.nonlinearity, dtype=dtype)
        else:
            raise NotImplementedError('Only LSTMCell, GRUCell and RNNCell             are supported for QuantizedRNN for now')
        assert mod.bias

        def _observe_and_quantize_weight(weight):
            if dtype == torch.qint8:
                weight_observer = weight_observer_method()
                weight_observer(weight)
                qweight = _quantize_weight(weight.float(), weight_observer)
                return qweight
            else:
                return weight.float()
        qRNNCellBase._packed_weight_ih = pack_weight_bias(_observe_and_quantize_weight(mod.weight_ih), mod.bias_ih, dtype)
        qRNNCellBase._packed_weight_hh = pack_weight_bias(_observe_and_quantize_weight(mod.weight_hh), mod.bias_hh, dtype)
        return qRNNCellBase

    @classmethod
    def from_reference(cls, ref_mod):
        assert hasattr(ref_mod, 'weight_ih_dtype'), 'We are assuming weight_ih '
        'exists in reference module, may need to relax the assumption to support the use case'
        if hasattr(ref_mod, 'nonlinearity'):
            qmod = cls(ref_mod.input_size, ref_mod.hidden_size, ref_mod.bias, ref_mod.nonlinearity, dtype=ref_mod.weight_ih_dtype)
        else:
            qmod = cls(ref_mod.input_size, ref_mod.hidden_size, ref_mod.bias, dtype=ref_mod.weight_ih_dtype)
        weight_bias_dict = {'weight': {'weight_ih': ref_mod.get_quantized_weight_ih(), 'weight_hh': ref_mod.get_quantized_weight_hh()}, 'bias': {'bias_ih': ref_mod.bias_ih, 'bias_hh': ref_mod.bias_hh}}
        qmod.set_weight_bias(weight_bias_dict)
        return qmod

    def _weight_bias(self):
        weight_bias_dict: Dict[str, Dict] = {'weight': {}, 'bias': {}}
        w1, b1 = self._packed_weight_ih.__getstate__()[0]
        w2, b2 = self._packed_weight_hh.__getstate__()[0]
        weight_bias_dict['weight']['weight_ih'] = w1
        weight_bias_dict['weight']['weight_hh'] = w2
        weight_bias_dict['bias']['bias_ih'] = b1
        weight_bias_dict['bias']['bias_hh'] = b2
        return weight_bias_dict

    def get_weight(self):
        return self._weight_bias()['weight']

    def get_bias(self):
        return self._weight_bias()['bias']

    def set_weight_bias(self, weight_bias_dict):
        self._packed_weight_ih = pack_weight_bias(weight_bias_dict['weight']['weight_ih'], weight_bias_dict['bias']['bias_ih'], self.weight_dtype)
        self._packed_weight_hh = pack_weight_bias(weight_bias_dict['weight']['weight_hh'], weight_bias_dict['bias']['bias_hh'], self.weight_dtype)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + '_packed_weight_ih'] = self._packed_weight_ih
        destination[prefix + '_packed_weight_hh'] = self._packed_weight_hh

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self._packed_weight_ih = state_dict.pop(prefix + '_packed_weight_ih')
        self._packed_weight_hh = state_dict.pop(prefix + '_packed_weight_hh')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs)