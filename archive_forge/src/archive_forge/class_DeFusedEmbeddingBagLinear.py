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
class DeFusedEmbeddingBagLinear(nn.Module):
    """A module to simulate QAT embedding bag with a linear layer,
    this module uses a separate embedding and bagging op, similar
    to that which is described in the EmbeddingBag documentation.

    https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html
    """

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=10, embedding_dim=12)
        self.emb.qconfig = default_embedding_qat_qconfig
        self.bagging_op = torch.sum
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.linear = nn.Linear(12, 1).to(dtype=torch.float)
        self.qconfig = get_default_qat_qconfig('qnnpack')

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.bagging_op(self.emb(input), dim=1)
        x = self.quant(x)
        x = self.linear(x)
        return self.dequant(x)