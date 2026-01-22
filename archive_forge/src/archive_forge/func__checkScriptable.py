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
def _checkScriptable(self, orig_mod, script_mod, calib_data, check_save_load):
    self._checkModuleCorrectnessAgainstOrig(orig_mod, script_mod, calib_data)
    buffer = io.BytesIO()
    torch.jit.save(script_mod, buffer)
    buffer.seek(0)
    loaded_mod = torch.jit.load(buffer)
    if check_save_load:
        self._checkModuleCorrectnessAgainstOrig(orig_mod, loaded_mod, calib_data)