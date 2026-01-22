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
class CriterionTest(InputVariableMixin, TestBase):
    _required_arg_names = TestBase._required_arg_names.union({'target'})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.should_test_cuda = kwargs.get('test_cuda', True)
        self.check_forward_only = kwargs.get('check_forward_only', False)
        self.check_gradgrad = kwargs.get('check_gradgrad', True)
        self.check_half = kwargs.get('check_half', True)
        self.check_bfloat16 = kwargs.get('check_bfloat16', False)
        self.check_complex = kwargs.get('check_complex', False)
        self.test_cpu = kwargs.get('test_cpu', True)
        self.with_tf32 = kwargs.get('with_tf32', True)
        self.tf32_precision = kwargs.get('tf32_precision', 0.001)
        self.check_batched_grad = kwargs.get('check_batched_grad', True)
        self.default_dtype = kwargs.get('default_dtype', None)
        if self.default_dtype is None:
            self.default_dtype = torch.get_default_dtype()

    def __call__(self, test_case):
        with set_default_dtype(self.default_dtype):
            module = self.constructor(*self.constructor_args)
            input = self._get_input()
            module.__repr__()
            str(module)
            target = self._get_target()
            if self.reference_fn is not None:
                out = test_case._forward_criterion(module, input, target, extra_args=self.extra_args)
                ref_args = (deepcopy(input), deepcopy(target)) + self.extra_args + (module,)
                expected_out = self.reference_fn(*ref_args)
                test_case.assertEqual(out, expected_out)
            if self.check_forward_only:
                return
            params = tuple((x for x in module.parameters()))
            if not isinstance(input, tuple):
                inputs = (input,) + params + (target,)

                def apply_fn(input, target, *params):
                    return module(input, target)
            else:
                inputs = input + params + (target,)

                def apply_fn(input1, input2, target, *params):
                    return module(input1, input2, target)
            gradcheck(apply_fn, inputs, check_batched_grad=self.check_batched_grad)
            if self.check_gradgrad:
                gradgradcheck(apply_fn, inputs, check_batched_grad=self.check_batched_grad)

    def test_cuda(self, test_case, dtype, extra_args=None):

        def convert_dtype(obj, dtype, requires_grad=False):
            if isinstance(obj, torch.Tensor):
                return obj.detach().to(dtype=dtype).requires_grad_(requires_grad)
            elif isinstance(obj, tuple):
                return tuple((convert_dtype(o, dtype, requires_grad) for o in obj))
            else:
                return obj
        if not TEST_CUDA or not self.should_test_cuda:
            raise unittest.SkipTest('Excluded from CUDA tests')
        with set_default_dtype(self.default_dtype):
            cpu_input = self._get_input()
            cpu_target = self._get_target()
            cpu_module = self.constructor(*self.constructor_args)
            gpu_module = self.constructor(*self.constructor_args)
            cpu_input = convert_dtype(cpu_input, dtype, True)
            if cpu_target.is_floating_point() or cpu_target.is_complex():
                cpu_target = convert_dtype(cpu_target, dtype)
            cpu_module.type(dtype)
            gpu_module.type(dtype)
            gpu_input = to_gpu(cpu_input)
            gpu_target = to_gpu(cpu_target)
            gpu_module.cuda()
            if dtype in {torch.half, torch.bfloat16}:
                cpu_input = self._get_input()
                cpu_target = self._get_target()
                cpu_module = self.constructor(*self.constructor_args)
            cpu_output = test_case._forward_criterion(cpu_module, cpu_input, cpu_target, extra_args=extra_args)
            gpu_output = test_case._forward_criterion(gpu_module, gpu_input, gpu_target, extra_args=extra_args)
            test_case.assertEqual(cpu_output, gpu_output, atol=0.1 if dtype in {torch.half, torch.bfloat16} else 0.0004, rtol=0, exact_dtype=False)
            cpu_gradInput = test_case._backward_criterion(cpu_module, cpu_input, cpu_output, cpu_target, extra_args=extra_args)
            gpu_gradInput = test_case._backward_criterion(gpu_module, gpu_input, gpu_output, gpu_target, extra_args=extra_args)
            test_case.assertEqual(cpu_gradInput, gpu_gradInput, atol=0.1 if dtype in {torch.half, torch.bfloat16} else 0.0004, rtol=0, exact_dtype=False)

    def _get_target(self):
        return self._get_arg('target', False)

    @property
    def constructor_args(self):
        return self._get_arg('constructor_args', False)

    @property
    def extra_args(self):
        return self._get_arg('extra_args', False)