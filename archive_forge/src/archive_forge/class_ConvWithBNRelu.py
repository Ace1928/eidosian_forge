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
class ConvWithBNRelu(torch.nn.Module):

    def __init__(self, relu, dim=2, bn=True, bias=True):
        super().__init__()
        convs = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d}
        bns = {1: torch.nn.BatchNorm1d, 2: torch.nn.BatchNorm2d}
        self.conv = convs[dim](3, 3, 3, bias=bias)
        if bn:
            self.bn = bns[dim](3)
        else:
            self.bn = torch.nn.Identity()
        if relu:
            self.relu = torch.nn.ReLU()
        else:
            self.relu = torch.nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)