import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.utils.fusion import fuse_linear_bn_weights
Create a qat module from a float module or qparams_dict

            Args: `mod' a float module, either produced by torch.ao.quantization
            utilities or directly from user
        