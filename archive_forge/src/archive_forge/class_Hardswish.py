import torch
from warnings import warn
class Hardswish(torch.nn.Hardswish):
    """This is the quantized version of :class:`~torch.nn.Hardswish`.

    Args:
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """

    def __init__(self, scale, zero_point, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.register_buffer('scale', torch.tensor(scale, **factory_kwargs))
        self.register_buffer('zero_point', torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        return torch.ops.quantized.hardswish(input, self.scale, self.zero_point)

    def _get_name(self):
        return 'QuantizedHardswish'

    @staticmethod
    def from_float(mod):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return Hardswish(float(scale), int(zero_point))

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(float(scale), int(zero_point))