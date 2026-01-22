import math
import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import fuse_conv_bn_weights
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.parameter import Parameter
from typing import TypeVar
class _ConvBnNd(nn.modules.conv._ConvNd, nni._FusedModule):
    _version = 2
    _FLOAT_MODULE = MOD

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, padding_mode, eps=1e-05, momentum=0.1, freeze_bn=False, qconfig=None, dim=2):
        nn.modules.conv._ConvNd.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, False, padding_mode)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.freeze_bn = freeze_bn if self.training else True
        self.bn = _BN_CLASS_MAP[dim](out_channels, eps, momentum, True, True)
        self.weight_fake_quant = self.qconfig.weight()
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_bn_parameters()
        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()
        self._enable_slow_path_for_better_numerical_stability = False

    def reset_running_stats(self):
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        self.bn.reset_running_stats()
        init.uniform_(self.bn.weight)
        init.zeros_(self.bn.bias)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def reset_parameters(self):
        super().reset_parameters()

    def update_bn_stats(self):
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        self.bn.training = False
        return self

    def _forward(self, input):
        if self._enable_slow_path_for_better_numerical_stability:
            return self._forward_slow(input)
        return self._forward_approximate(input)

    def _forward_approximate(self, input):
        """Approximated method to fuse conv and bn. It requires only one forward pass.
        conv_orig = conv / scale_factor where scale_factor = bn.weight / running_std
        """
        assert self.bn.running_var is not None
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias, dtype=input.dtype)
        else:
            zero_bias = torch.zeros(self.out_channels, device=scaled_weight.device, dtype=input.dtype)
        conv = self._conv_forward(input, scaled_weight, zero_bias)
        conv_orig = conv / scale_factor.reshape(bias_shape)
        if self.bias is not None:
            conv_orig = conv_orig + self.bias.reshape(bias_shape)
        conv = self.bn(conv_orig)
        return conv

    def _forward_slow(self, input):
        """
        A more accurate but slow method to compute conv bn fusion, following https://arxiv.org/pdf/1806.08342.pdf
        It requires two forward passes but handles the case bn.weight == 0

        Conv: Y = WX + B_c
        Conv without bias: Y0 = WX = Y - B_c, Y = Y0 + B_c

        Batch statistics:
          mean_Y = Y.mean()
                 = Y0.mean() + B_c
          var_Y = (Y - mean_Y)^2.mean()
                = (Y0 - Y0.mean())^2.mean()
        BN (r: bn.weight, beta: bn.bias):
          Z = r * (Y - mean_Y) / sqrt(var_Y + eps) + beta
            = r * (Y0 - Y0.mean()) / sqrt(var_Y + eps) + beta

        Fused Conv BN training (std_Y = sqrt(var_Y + eps)):
          Z = (r * W / std_Y) * X + r * (B_c - mean_Y) / std_Y + beta
            = (r * W / std_Y) * X - r * Y0.mean() / std_Y + beta

        Fused Conv BN inference (running_std = sqrt(running_var + eps)):
          Z = (r * W / running_std) * X - r * (running_mean - B_c) / running_std + beta

        QAT with fused conv bn:
          Z_train = fake_quant(r * W / running_std) * X * (running_std / std_Y) - r * Y0.mean() / std_Y + beta
                  = conv(X, fake_quant(r * W / running_std)) * (running_std / std_Y) - r * Y0.mean() / std_Y + beta
          Z_inference = conv(X, fake_quant(r * W / running_std)) - r * (running_mean - B_c) / running_std + beta
        """
        assert self.bn.running_var is not None
        assert self.bn.running_mean is not None
        zero_bias = torch.zeros(self.out_channels, device=self.weight.device, dtype=input.dtype)
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        if self.bn.training:
            conv_out = self._conv_forward(input, self.weight, zero_bias)
            with torch.no_grad():
                conv_out_bias = conv_out if self.bias is None else conv_out + self.bias.reshape(bias_shape)
                self.bn(conv_out_bias)
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
        conv_bn = self._conv_forward(input, scaled_weight, zero_bias)
        if self.bn.training:
            avg_dims = [0] + list(range(2, len(self.weight.shape)))
            batch_mean = conv_out.mean(avg_dims)
            batch_var = torch.square(conv_out - batch_mean.reshape(bias_shape)).mean(avg_dims)
            batch_std = torch.sqrt(batch_var + self.bn.eps)
            unscale_factor = running_std / batch_std
            conv_bn *= unscale_factor.reshape(bias_shape)
            fused_mean = batch_mean
            fused_std = batch_std
        else:
            fused_mean = self.bn.running_mean - (self.bias if self.bias is not None else 0)
            fused_std = running_std
        fused_bias = self.bn.bias - self.bn.weight * fused_mean / fused_std
        conv_bn += fused_bias.reshape(bias_shape)
        if self.bias is not None:
            conv_bn += (self.bias - self.bias).reshape(bias_shape)
        return conv_bn

    def extra_repr(self):
        return super().extra_repr()

    def forward(self, input):
        return self._forward(input)

    def train(self, mode=True):
        """
        Batchnorm's training behavior is using the self.training flag. Prevent
        changing it if BN is frozen. This makes sure that calling `model.train()`
        on a model with a frozen BN will behave properly.
        """
        self.training = mode
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version == 1:
            v2_to_v1_names = {'bn.weight': 'gamma', 'bn.bias': 'beta', 'bn.running_mean': 'running_mean', 'bn.running_var': 'running_var', 'bn.num_batches_tracked': 'num_batches_tracked'}
            for v2_name, v1_name in v2_to_v1_names.items():
                if prefix + v1_name in state_dict:
                    state_dict[prefix + v2_name] = state_dict[prefix + v1_name]
                    state_dict.pop(prefix + v1_name)
                elif prefix + v2_name in state_dict:
                    pass
                elif strict:
                    missing_keys.append(prefix + v2_name)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @classmethod
    def from_float(cls, mod):
        """Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.ao.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        qconfig = mod.qconfig
        conv, bn = (mod[0], mod[1])
        qat_convbn = cls(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups, conv.bias is not None, conv.padding_mode, bn.eps, bn.momentum, False, qconfig)
        qat_convbn.weight = conv.weight
        qat_convbn.bias = conv.bias
        qat_convbn.bn.weight = bn.weight
        qat_convbn.bn.bias = bn.bias
        qat_convbn.bn.running_mean = bn.running_mean
        qat_convbn.bn.running_var = bn.running_var
        qat_convbn.bn.num_batches_tracked = bn.num_batches_tracked
        return qat_convbn

    def to_float(self):
        cls = type(self)
        conv = cls._FLOAT_CONV_MODULE(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias is not None, self.padding_mode)
        conv.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            conv.bias = torch.nn.Parameter(self.bias.detach())
        if cls._FLOAT_BN_MODULE:
            assert self.bn.running_var is not None and self.bn.running_mean is not None
            conv.weight, conv.bias = fuse_conv_bn_weights(conv.weight, conv.bias, self.bn.running_mean, self.bn.running_var, self.bn.eps, self.bn.weight, self.bn.bias)
        if cls._FLOAT_RELU_MODULE:
            modules = []
            modules.append(conv)
            relu = cls._FLOAT_RELU_MODULE()
            modules.append(relu)
            conv_relu = cls._FUSED_FLOAT_MODULE(*modules)
            conv_relu.train(self.training)
            return conv_relu
        else:
            conv.train(self.training)
            return conv