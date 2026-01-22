import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
from torch.ao.nn.intrinsic import _FusedModule
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM
from torch.ao.quantization import (
from torch.ao.quantization import QuantWrapper, QuantStub, DeQuantStub, \
from torch.ao.quantization.quantization_mappings import (
from torch.testing._internal.common_quantized import (
from torch.jit.mobile import _load_for_lite_interpreter
import copy
import io
import functools
import time
import os
import unittest
import numpy as np
from torch.testing import FileCheck
from typing import Callable, Tuple, Dict, Any, Union, Type, Optional
import torch._dynamo as torchdynamo
class ModelForConvTransposeBNFusion(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(3, 3, 1)
        self.bn1 = nn.BatchNorm1d(3)
        self.conv2 = nn.ConvTranspose2d(3, 3, 1)
        self.bn2 = nn.BatchNorm2d(3)
        self.conv3 = nn.ConvTranspose3d(3, 3, 1)
        self.bn3 = nn.BatchNorm3d(3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = x.unsqueeze(2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x.unsqueeze(2)
        x = self.conv3(x)
        x = self.bn3(x)
        return x