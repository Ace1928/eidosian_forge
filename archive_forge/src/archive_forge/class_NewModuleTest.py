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
class NewModuleTest(InputVariableMixin, ModuleTest):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cudnn = kwargs.get('cudnn', False)
        self.check_inplace = kwargs.get('check_inplace', False)
        self.check_gradgrad = kwargs.get('check_gradgrad', True)
        self.skip_double = kwargs.get('skip_double', False)
        self.skip_half = kwargs.get('skip_half', False)
        self.with_tf32 = kwargs.get('with_tf32', False)
        self.tf32_precision = kwargs.get('tf32_precision', 0.001)
        self.test_cpu = kwargs.get('test_cpu', True)
        self.has_sparse_gradients = kwargs.get('has_sparse_gradients', False)
        self.check_batched_grad = kwargs.get('check_batched_grad', True)
        self.gradcheck_fast_mode = kwargs.get('gradcheck_fast_mode', None)
        self.supports_forward_ad = kwargs.get('supports_forward_ad', False)
        self.supports_fwgrad_bwgrad = kwargs.get('supports_fwgrad_bwgrad', False)

    def _check_gradients(self, test_case, module, input_tuple):
        params = tuple((x for x in module.parameters()))
        num_inputs = len(input_tuple)

        def fn_to_gradcheck(*inputs_and_params, **kwargs):
            assert not kwargs
            return test_case._forward(module, inputs_and_params[:num_inputs])
        if self.has_sparse_gradients:
            assert num_inputs == 1
            test_input_jacobian = torch.is_floating_point(input_tuple[0])
            test_case.check_jacobian(module, input_tuple[0], test_input_jacobian)
        else:
            test_case.assertTrue(gradcheck(fn_to_gradcheck, input_tuple + params, check_batched_grad=self.check_batched_grad, fast_mode=self.gradcheck_fast_mode, check_forward_ad=self.supports_forward_ad))
        if self.check_gradgrad:
            test_case.assertTrue(gradgradcheck(fn_to_gradcheck, input_tuple + params, check_batched_grad=self.check_batched_grad, fast_mode=self.gradcheck_fast_mode, check_fwd_over_rev=self.supports_fwgrad_bwgrad))

    def _do_test(self, test_case, module, input):
        num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        input_tuple = input if isinstance(input, tuple) else (input,)
        self._check_gradients(test_case, module, input_tuple)
        module.__repr__()
        if self.check_inplace:
            assert len(input_tuple) == 1
            input = input_tuple[0]
            module_ip = self.constructor(*self.constructor_args, inplace=True)
            input_version = input._version
            with freeze_rng_state():
                output = module(input)
            test_case.assertEqual(input._version, input_version)
            input_ip = deepcopy(input)
            input_ip_clone = input_ip.clone()
            with freeze_rng_state():
                output_ip = module_ip(input_ip_clone)
            test_case.assertNotEqual(input_ip_clone._version, input_version)
            test_case.assertEqual(output, output_ip)
            grad = output.data.clone().normal_()
            if input.grad is not None:
                with torch.no_grad():
                    input.grad.zero_()
            if input_ip.grad is not None:
                with torch.no_grad():
                    input_ip.grad.zero_()
            output.backward(grad)
            output_ip.backward(grad)
            test_case.assertEqual(input.grad, input_ip.grad)

        def assert_module_parameters_are(tensor_type, device_id=None):
            for p in module.parameters():
                test_case.assertIsInstance(p, tensor_type)
                if device_id is not None:
                    test_case.assertEqual(p.get_device(), device_id)
        if all((isinstance(t, torch.LongTensor) for t in input_tuple)) and TEST_CUDA:
            input_tuple = tuple((t.cuda() for t in input_tuple))
            module.float().cuda()
            module(*input_tuple)
            assert_module_parameters_are(torch.cuda.FloatTensor, 0)
            if torch.cuda.device_count() > 1:
                input_tuple = tuple((t.cuda(1) for t in input_tuple))
                module.cuda(1)
                with torch.cuda.device(1):
                    module(*input_tuple)
                assert_module_parameters_are(torch.cuda.FloatTensor, 1)
        else:

            def to_type(tensor, real, complex):
                if tensor.is_complex():
                    return tensor.to(complex)
                elif tensor.is_floating_point():
                    return tensor.to(real)
                else:
                    return tensor

            def to_half(x):
                return to_type(x, torch.float16, None)

            def to_single(x):
                return to_type(x, torch.float32, torch.complex64)

            def to_double(x):
                return to_type(x, torch.float64, torch.complex128)
            input_tuple = tuple((to_single(t) for t in input_tuple))
            module.float()
            module(*input_tuple)
            assert_module_parameters_are(torch.FloatTensor)
            input_tuple = tuple((to_double(t) for t in input_tuple))
            module.double()
            module(*input_tuple)
            assert_module_parameters_are(torch.DoubleTensor)
            if TEST_CUDA and self.should_test_cuda:
                input_tuple = tuple((to_single(t).cuda() for t in input_tuple))
                module.float().cuda()
                module(*input_tuple)
                assert_module_parameters_are(torch.cuda.FloatTensor, 0)
                input_tuple = tuple((t.cpu() for t in input_tuple))
                module.cpu()
                module(*input_tuple)
                assert_module_parameters_are(torch.FloatTensor)
                input_tuple = tuple((t.cuda() for t in input_tuple))
                module.cuda()
                module(*input_tuple)
                assert_module_parameters_are(torch.cuda.FloatTensor, 0)
                if self.cudnn:
                    with torch.backends.cudnn.flags(enabled=False):
                        module(*input_tuple)
                        assert_module_parameters_are(torch.cuda.FloatTensor, 0)
                if torch.cuda.device_count() >= 2:
                    input_tuple = tuple((t.cuda(1) for t in input_tuple))
                    module.cuda(1)
                    with torch.cuda.device(1):
                        module(*input_tuple)
                    assert_module_parameters_are(torch.cuda.FloatTensor, 1)
                if not self.skip_double:
                    input_tuple = tuple((to_double(t).cuda() for t in input_tuple))
                    module.double().cuda()
                    module(*input_tuple)
                    assert_module_parameters_are(torch.cuda.DoubleTensor, 0)
                if not self.skip_half:
                    input_tuple = tuple((to_half(t).cuda() for t in input_tuple))
                    module.half().cuda()
                    module(*input_tuple)
                    assert_module_parameters_are(torch.cuda.HalfTensor, 0)
        torch.set_num_threads(num_threads)

    def _get_target(self):
        return self._get_arg('target', False)

    @property
    def constructor_args(self):
        return self._get_arg('constructor_args', False)