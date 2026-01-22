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
class EmbeddingWithStaticLinear(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12)
        self.fc = torch.nn.Linear(4, 2)
        self.emb.qconfig = float_qparams_weight_only_qconfig
        self.qconfig = default_qconfig
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, indices, offsets, linear_in):
        emb = self.emb(indices, offsets)
        q_x = self.quant(linear_in)
        fc = self.fc(q_x)
        fc = self.dequant(fc)
        features = torch.cat([fc] + [emb], dim=1)
        return features