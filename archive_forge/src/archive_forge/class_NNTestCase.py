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
class NNTestCase(TestCase):

    @abstractmethod
    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_parameters(self, module: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        raise NotImplementedError

    @abstractmethod
    def _zero_grad_parameters(self, module: nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def _backward(self, module: nn.Module, input: _TensorOrTensors, output: torch.Tensor, grad_output: Union[torch.Tensor, Sequence[torch.Tensor]], create_graph: bool=False):
        raise NotImplementedError

    def _jacobian(self, input, num_out):
        if isinstance(input, tuple):
            return tuple((self._jacobian(elem, num_out) for elem in input))
        elif isinstance(input, list):
            return [self._jacobian(elem, num_out) for elem in input]
        else:
            return torch.zeros(input.nelement(), num_out)

    def _flatten_tensors(self, x):
        if isinstance(x, torch.Tensor):
            if x.is_sparse:
                return x.to_dense().view(-1)
            else:
                return x.view(-1)
        else:
            return tuple((self._flatten_tensors(a) for a in x))

    def _zero_grad_input(self, input):
        if isinstance(input, torch.Tensor):
            if input.requires_grad and input.grad is not None:
                input.grad.zero_()
                input.grad.detach_()
        else:
            for i in input:
                self._zero_grad_input(i)

    def _analytical_jacobian(self, module, input: _TensorOrTensors, jacobian_input=True, jacobian_parameters=True):
        output = self._forward(module, input)
        output_size = output.nelement()
        if jacobian_input:
            jacobian_inp = self._jacobian(input, output_size)
            flat_jacobian_input = list(_iter_tensors(jacobian_inp))
        if jacobian_parameters:
            num_param = sum((p.numel() for p in self._get_parameters(module)[0]))
            jacobian_param = torch.zeros(num_param, output_size)
        for i in range(output_size):
            param, d_param = self._get_parameters(module)
            d_param = [torch.zeros_like(p) if d is None else d for p, d in zip(param, d_param)]
            d_out = torch.zeros_like(output)
            flat_d_out = d_out.view(-1)
            flat_d_out[i] = 1
            if jacobian_parameters:
                self._zero_grad_parameters(module)
            if jacobian_input:
                self._zero_grad_input(input)
            d_input = self._backward(module, input, output, d_out)
            if jacobian_input:
                for jacobian_x, d_x in zip(flat_jacobian_input, _iter_tensors(d_input)):
                    jacobian_x[:, i] = d_x.contiguous().view(-1)
            if jacobian_parameters:
                jacobian_param[:, i] = torch.cat(self._flatten_tensors(d_param), 0)
        res: Tuple[torch.Tensor, ...] = tuple()
        if jacobian_input:
            res += (jacobian_inp,)
        if jacobian_parameters:
            res += (jacobian_param,)
        return res

    def _numerical_jacobian(self, module, input: _TensorOrTensors, jacobian_input=True, jacobian_parameters=True):

        def fw(*input):
            return self._forward(module, input).detach()
        res: Tuple[torch.Tensor, ...] = tuple()
        if jacobian_input:
            res += (_get_numerical_jacobian(fw, input, eps=1e-06),)
        if jacobian_parameters:
            param, _ = self._get_parameters(module)
            to_cat = []
            for p in param:
                jacobian = _get_numerical_jacobian(fw, input, target=p, eps=1e-06)
                to_cat.append(jacobian[0][0])
            res += (torch.cat(to_cat, 0),)
        return res

    def check_jacobian(self, module, input: _TensorOrTensors, jacobian_input=True):
        jacobian_parameters = bool(self._get_parameters(module)[0])
        analytical = self._analytical_jacobian(module, input, jacobian_input, jacobian_parameters)
        numerical = self._numerical_jacobian(module, input, jacobian_input, jacobian_parameters)
        analytical_t = list(_iter_tensors(analytical))
        numerical_t = list(_iter_tensors(numerical))
        differences = []
        for a, n in zip(analytical_t, numerical_t):
            if a.numel() != 0:
                differences.append(a.add(n, alpha=-1).abs().max())
        if len(differences) > 0:
            self.assertLessEqual(max(differences), PRECISION)