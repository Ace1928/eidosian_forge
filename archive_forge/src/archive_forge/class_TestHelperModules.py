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
class TestHelperModules:

    class Conv2dPropAnnotaton(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(-1, 3)
            x = torch.nn.functional.hardtanh(x, -0.5, 0.5)
            x = self.linear(x)
            return x

    class Conv2dWithObsSharingOps(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.hardtanh = torch.nn.Hardtanh()
            self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv(x)
            x = self.adaptive_avg_pool2d(x)
            x = self.hardtanh(x)
            x = torch.mean(x)
            return x

    class Conv2dWithTwoLinearPermute(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3)
            self.linear1 = torch.nn.Linear(16, 8, bias=False)
            self.linear2 = torch.nn.Linear(8, 8)

        def forward(self, x):
            conv_out = self.conv(x)
            permute_out = torch.permute(conv_out, (0, 2, 3, 1))
            return self.linear2(self.linear1(permute_out))

    class Conv2dWithTwoLinear(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3)
            self.linear1 = torch.nn.Linear(64, 8, bias=False)
            self.linear2 = torch.nn.Linear(8, 8)

        def forward(self, x):
            conv_out = self.conv(x)
            reshape_out = torch.reshape(conv_out, (2, 64))
            return self.linear2(self.linear1(reshape_out))

    class ConvLinearWPermute(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 3)
            self.linear1 = torch.nn.Linear(8, 8)

        def forward(self, x):
            conv_out = self.conv(x)
            permute_out = torch.permute(conv_out, (0, 2, 3, 1))
            return self.linear1(permute_out)

    class TwoLinearModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(8, 16, bias=False)
            self.linear2 = torch.nn.Linear(16, 8)

        def forward(self, x):
            return self.linear2(self.linear1(x))

    class ConvMaxPool2d(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(2, 2, 1)
            self.pool = torch.nn.MaxPool2d(1, 1)

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            return x

    class ConvWithAdaptiveAvgPool2d(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv(x)
            x = self.adaptive_avg_pool2d(x)
            return x

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

    class Conv2dThenConv1d(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1d = torch.nn.Conv1d(3, 3, 3)
            self.conv2d = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            x = self.conv2d(x)
            x = x.squeeze(0)
            x = self.conv1d(x)
            return x

        def example_inputs(self):
            return (torch.randn(1, 3, 5, 5),)

    class Conv2dWithCat(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.conv2 = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x, y):
            x = self.conv1(x)
            y = self.conv2(y)
            z = torch.cat([x, y], dim=1)
            return z

    class Conv2dWithTwoCat(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.conv2 = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x1, x2, x3, x4):
            x1 = self.conv1(x1)
            x2 = self.conv2(x2)
            y = torch.cat([x1, x2], dim=1)
            z = x3 + x4
            w = torch.cat([z, y])
            return w

    class ThreeAdd(torch.nn.Module):

        def forward(self, x1, x2, x3, x4):
            y = x1 + x2
            z = x3 + x4
            w = y + z
            return w

    class EmbeddingModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)

        def forward(self, indices):
            return self.emb(indices)

    class EmbeddingConvLinearModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=8)
            self.conv = torch.nn.Conv2d(8, 16, (1, 3))
            self.linear = torch.nn.Linear(16, 8)

        def forward(self, indices):
            embeddings = self.emb(indices)
            embeddings = torch.unsqueeze(embeddings, dim=0)
            embeddings = torch.permute(embeddings, (0, 3, 1, 2))
            conv_out = self.conv(embeddings)
            conv_out = torch.permute(conv_out, (0, 2, 3, 1))
            conv_out = torch.squeeze(conv_out, dim=0)
            return self.linear(conv_out)

    class AddInplaceAdd(torch.nn.Module):

        def forward(self, x, y):
            x = x + y
            x += y
            return x

    class MulInplaceMul(torch.nn.Module):

        def forward(self, x, y):
            x = x * y
            x *= y
            return x

    class AddMulScalar(torch.nn.Module):

        def forward(self, x):
            x = x + 3
            x = x * 3
            x += 3
            x *= 3
            return x

    class ConvBnReLU2dAndLinearReLU(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv_bn_relu = TestHelperModules.ConvWithBNRelu(relu=True)
            self.linear = torch.nn.Linear(3, 8, bias=False)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.conv_bn_relu(x)
            permute_out = torch.permute(x, (0, 2, 3, 1))
            linear_out = self.linear(permute_out)
            return linear_out