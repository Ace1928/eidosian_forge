import torch
import torch.ao.nn.intrinsic
import torch.ao.nn.intrinsic.qat
import torch.nn.functional as F
import torch.ao.nn.quantized as nnq

    A ConvAddReLU2d module is a fused module of Conv2d, Add and Relu

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv2d

    