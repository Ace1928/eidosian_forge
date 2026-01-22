import torch
from torch.nn import Module
from torch.ao.quantization.observer import (
import re
from abc import ABC, abstractmethod
from typing import Any, Tuple
class FakeQuantize(FakeQuantizeBase):
    """Simulate the quantize and dequantize operations in training time.

    The output of this module is given by::

        x_out = (
          clamp(round(x/scale + zero_point), quant_min, quant_max) - zero_point
        ) * scale

    * :attr:`is_dynamic` indicates whether the fake quantie is a placeholder for dynamic quantization
      operators (choose_qparams -> q -> dq) or static quantization operators (q -> dq)

    * :attr:`scale` defines the scale factor used for quantization.

    * :attr:`zero_point` specifies the quantized value to which 0 in floating point maps to

    * :attr:`fake_quant_enabled` controls the application of fake quantization on tensors, note that
      statistics can still be updated.

    * :attr:`observer_enabled` controls statistics collection on tensors

    * :attr:`dtype` specifies the quantized dtype that is being emulated with fake-quantization,
        allowable values are torch.qint8 and torch.quint8.

    Args:

        observer (module): Module for observing statistics on input tensors and calculating scale
          and zero-point.
        observer_kwargs (optional): Arguments for the observer module

    Attributes:
        activation_post_process (Module): User provided module that collects statistics on the input tensor and
          provides a method to calculate scale and zero-point.

    """
    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=None, quant_max=None, is_dynamic=False, **observer_kwargs):
        super().__init__()
        if quant_min is not None and quant_max is not None:
            assert quant_min <= quant_max, 'quant_min must be less than or equal to quant_max'
            dtype = observer_kwargs.get('dtype', torch.quint8)
            if hasattr(observer, 'p'):
                dtype = getattr(getattr(observer, 'p', {}), 'keywords', {}).get('dtype', dtype)
            assert torch.iinfo(dtype).min <= quant_min, 'quant_min out of bound'
            assert quant_max <= torch.iinfo(dtype).max, 'quant_max out of bound'
            observer_kwargs.update({'quant_min': quant_min, 'quant_max': quant_max})
        observer_kwargs['is_dynamic'] = is_dynamic
        self.activation_post_process = observer(**observer_kwargs)
        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max
        self.is_dynamic = self.activation_post_process.is_dynamic
        if _is_float_qparams(self.activation_post_process.qscheme):
            zero_point_dtype = torch.float
        else:
            zero_point_dtype = torch.int
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=zero_point_dtype))
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = self.activation_post_process.ch_axis if hasattr(self.activation_post_process, 'ch_axis') else -1
        assert _is_per_channel(self.qscheme) or _is_per_tensor(self.qscheme), 'Only per channel and per tensor quantization are supported in fake quantize' + ' got qscheme: ' + str(self.qscheme)
        self.is_per_channel = _is_per_channel(self.qscheme)

    @torch.jit.export
    def calculate_qparams(self):
        return self.activation_post_process.calculate_qparams()

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = (_scale.to(self.scale.device), _zero_point.to(self.zero_point.device))
            if self.scale.shape != _scale.shape:
                self.scale.resize_(_scale.shape)
                self.zero_point.resize_(_zero_point.shape)
            self.scale.copy_(_scale)
            self.zero_point.copy_(_zero_point)
        if self.fake_quant_enabled[0] == 1:
            if self.is_per_channel:
                X = torch.fake_quantize_per_channel_affine(X, self.scale, self.zero_point, self.ch_axis, self.activation_post_process.quant_min, self.activation_post_process.quant_max)
            else:
                X = torch.fake_quantize_per_tensor_affine(X, self.scale, self.zero_point, self.activation_post_process.quant_min, self.activation_post_process.quant_max)
        return X

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, scale={}, zero_point={}'.format(self.fake_quant_enabled, self.observer_enabled, self.activation_post_process.quant_min, self.activation_post_process.quant_max, self.dtype, self.qscheme, self.ch_axis, self.scale, self.zero_point)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                if name == 'scale':
                    self.scale.resize_(val.shape)
                else:
                    assert name == 'zero_point'
                    self.zero_point.resize_(val.shape)
                if torch.jit.is_scripting():
                    if name == 'scale':
                        self.scale.copy_(val)
                    else:
                        assert name == 'zero_point'
                        self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)