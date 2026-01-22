import torch
class InstanceNorm1d(torch.nn.InstanceNorm1d):
    """This is the quantized version of :class:`~torch.nn.InstanceNorm1d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """

    def __init__(self, num_features, weight, bias, scale, zero_point, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(num_features, eps, momentum, affine, track_running_stats, **factory_kwargs)
        self.weight = weight
        self.bias = bias
        self.register_buffer('scale', torch.tensor(scale, **factory_kwargs))
        self.register_buffer('zero_point', torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        return torch.ops.quantized.instance_norm(input, self.weight, self.bias, self.eps, self.scale, self.zero_point)

    def _get_name(self):
        return 'QuantizedInstanceNorm1d'

    @classmethod
    def from_float(cls, mod):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(mod.num_features, mod.weight, mod.bias, float(scale), int(zero_point), mod.eps, mod.affine)
        return new_mod

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(mod.num_features, mod.weight, mod.bias, float(scale), int(zero_point), mod.eps, mod.affine)