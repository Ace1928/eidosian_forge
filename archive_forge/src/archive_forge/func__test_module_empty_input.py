from abc import abstractmethod
import tempfile
import unittest
from copy import deepcopy
from functools import reduce, partial, wraps
from itertools import product
from operator import mul
from math import pi
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import _reduction as _Reduction
from torch.testing._internal.common_utils import TestCase, to_gpu, freeze_rng_state, is_iterable, \
from torch.testing._internal.common_cuda import TEST_CUDA, SM90OrLater
from torch.autograd.gradcheck import _get_numerical_jacobian, _iter_tensors
from torch.autograd import Variable
from torch.types import _TensorOrTensors
import torch.backends.cudnn
from typing import Dict, Callable, Tuple, List, Sequence, Union, Any
def _test_module_empty_input(test_case, module, inp, check_size=True, inference=False):
    if not inference:
        inp.requires_grad_(True)
    out = module(inp)
    if not inference:
        gO = torch.rand_like(out)
        out.backward(gO)
    if check_size:
        test_case.assertEqual(out.size(), inp.size())
    if not inference:
        for p in module.parameters():
            if p.requires_grad:
                test_case.assertEqual(p.grad, torch.zeros_like(p.grad))
        test_case.assertEqual(inp.grad, torch.zeros_like(inp))