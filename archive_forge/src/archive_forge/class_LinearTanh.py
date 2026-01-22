import torch
import torch.ao.nn.quantized as nnq
import torch.ao.nn.intrinsic as nni
from torch.ao.nn.quantized.modules.utils import _quantize_weight
class LinearTanh(nnq.Linear):
    """
    A LinearTanh module fused from Linear and Tanh modules

    We adopt the same interface as :class:`torch.ao.nn.quantized.Linear`.

    Attributes:
        Same as torch.ao.nn.quantized.Linear

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.LinearTanh(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    _FLOAT_MODULE = nni.LinearTanh

    def __init__(self, in_features, out_features, bias=True, dtype=torch.qint8):
        super().__init__(in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.linear_tanh(x, self._packed_params._packed_params, self.scale, self.zero_point)

    def _get_name(self):
        return 'QuantizedLinearTanh'

    @classmethod
    def from_float(cls, mod):
        assert type(mod) == nni.LinearTanh, 'Input float module should be LinearTanh'
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        activation_post_process = mod.activation_post_process
        mod = mod[0]
        weight_post_process = mod.qconfig.weight()
        weight_post_process(mod.weight)
        dtype = weight_post_process.dtype
        act_scale, act_zp = activation_post_process.calculate_qparams()
        assert dtype == torch.qint8, 'Weight observer must have dtype torch.qint8'
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        qlinear_tanh = cls(mod.in_features, mod.out_features, dtype=dtype)
        qlinear_tanh.set_weight_bias(qweight, mod.bias)
        qlinear_tanh.scale = float(act_scale)
        qlinear_tanh.zero_point = int(act_zp)
        return qlinear_tanh

    @classmethod
    def from_reference(cls, ref_mod, output_scale, output_zero_point):
        linear = ref_mod[0]
        qlinear_tanh = cls(linear.in_features, linear.out_features)
        qweight = linear.get_quantized_weight()
        qlinear_tanh.set_weight_bias(qweight, linear.bias)
        qlinear_tanh.scale = float(output_scale)
        qlinear_tanh.zero_point = int(output_zero_point)
        return qlinear_tanh