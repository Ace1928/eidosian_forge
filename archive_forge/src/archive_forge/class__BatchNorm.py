import torch
import torch.ao.nn.intrinsic as nni
class _BatchNorm(torch.nn.modules.batchnorm._BatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(num_features, eps, momentum, True, True, **factory_kwargs)
        self.register_buffer('scale', torch.tensor(1.0, **factory_kwargs))
        self.register_buffer('zero_point', torch.tensor(0, **factory_kwargs))

    @staticmethod
    def from_float(cls, mod):
        activation_post_process = mod.activation_post_process
        if type(mod) == cls._NNI_BN_RELU_MODULE:
            mod = mod[0]
        scale, zero_point = activation_post_process.calculate_qparams()
        new_mod = cls(mod.num_features, mod.eps)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.running_mean = mod.running_mean
        new_mod.running_var = mod.running_var
        new_mod.scale = scale
        new_mod.zero_point = zero_point
        return new_mod

    @classmethod
    def from_reference(cls, bn, output_scale, output_zero_point):
        qbn = cls(bn.num_features, bn.eps, bn.momentum, device=bn.weight.device, dtype=bn.weight.dtype)
        qbn.weight = bn.weight
        qbn.bias = bn.bias
        qbn.running_mean = bn.running_mean
        qbn.running_var = bn.running_var
        qbn.scale = output_scale
        qbn.zero_point = output_zero_point
        return qbn