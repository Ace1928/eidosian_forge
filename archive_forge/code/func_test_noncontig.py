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
def test_noncontig(self, test_case, module, input):
    if isinstance(input, torch.Tensor) and input.dim() == 0:
        return
    if any((i.dim() == 0 for i in input if isinstance(i, torch.Tensor))):
        return
    test_case._zero_grad_parameters(module)
    test_case._zero_grad_input(input)
    with freeze_rng_state():
        output = test_case._forward(module, input)
        if getattr(module, 'return_indices', False):
            output = output[0]
        grad_output = output.new(output.shape).normal_()
        output = output.clone()
        d_input = deepcopy(test_case._backward(module, input, output, grad_output))
        d_param = deepcopy(test_case._get_parameters(module)[1])
    nc_input = self.noncontiguize(input)
    nc_grad_output = self.noncontiguize(grad_output)
    for contig_i, contig_g in product((True, False), repeat=2):
        i = input if contig_i else nc_input
        go = deepcopy(grad_output if contig_g else nc_grad_output)
        test_case._zero_grad_parameters(module)
        test_case._zero_grad_input(i)
        with freeze_rng_state():
            out = test_case._forward(module, i)
            if getattr(module, 'return_indices', False):
                out = out[0]
            grad = test_case._backward(module, i, out, go)
            test_case.assertEqual(out, output)
            test_case.assertEqual(grad, d_input, atol=0.0001, rtol=0)
            test_case.assertEqual(test_case._get_parameters(module)[1], d_param)