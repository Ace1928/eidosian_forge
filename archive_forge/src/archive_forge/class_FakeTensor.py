import contextlib
import functools
import itertools
import logging
import os
import sys
import traceback
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from weakref import ReferenceType
import torch
import torch._custom_op
import torch._logging
from torch._guards import Source
from torch._ops import OpOverload
from torch._prims_common import (
from torch._subclasses.meta_utils import MetaConverter
from torch._utils import render_call
from torch.fx.operator_schemas import normalize_function
from torch.multiprocessing.reductions import StorageWeakRef
from torch.overrides import TorchFunctionMode
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import (
from torch.utils._pytree import PyTree, tree_map
from torch.utils._stats import count, count_label
from torch.utils.weak import WeakIdRef
class FakeTensor(torch.Tensor):
    """
    Meta tensors give you the ability to run PyTorch code without having to
    actually do computation through tensors allocated on a `meta` device.
    Because the device is `meta`, meta tensors do not model device propagation.
    FakeTensor extends MetaTensors to also carry an additional `fake_device`
    which tracks devices that would have been used.
    """
    fake_device: torch.device
    fake_mode: 'FakeTensorMode'
    constant: Optional[torch.Tensor]
    _nonzero_memo: Optional[torch.SymInt]
    _nonzero_memo_vc: Optional[int]
    _mode_key = torch._C._TorchDispatchModeKey.FAKE

    @property
    def nonzero_memo(self):
        if self._nonzero_memo is None:
            return None
        if self._nonzero_memo_vc != self._version:
            self._nonzero_memo = None
            return None
        return self._nonzero_memo

    @property
    def device(self):
        if self.fake_mode.in_kernel_invocation:
            return torch.device('meta')
        else:
            return self.fake_device

    @staticmethod
    def __new__(cls, fake_mode, elem, device, constant=None):
        self = torch.Tensor._make_subclass(cls, elem, elem.requires_grad, dispatch_device=True, device_for_backend_keys=device)
        assert elem.device.type == 'meta', elem.device.type
        device = device if isinstance(device, torch.device) else torch.device(device)
        if not fake_mode.allow_meta:
            assert device.type != 'meta'
        if device.type == 'cuda':
            init_cuda_context()
        if device.type in ['cuda', 'hpu', 'xpu', torch._C._get_privateuse1_backend_name()] and device.index is None:
            device = torch.device(f'{device.type}:{getattr(torch, device.type).current_device()}')
        self.fake_device = device
        self.fake_mode = fake_mode
        self.constant = constant
        self._nonzero_memo = None
        self._nonzero_memo_vc = None
        if FakeTensorConfig.debug:
            import traceback
            self._debug_trace = traceback.extract_stack()
        return self

    def __init__(self, *args, **kwargs):
        super().__init__()

    @staticmethod
    def from_tensor(t, fake_mode):
        return fake_mode.from_tensor(t)

    @classmethod
    @count
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func == torch.ops.prim.device.default:
            assert len(args) == 1 and isinstance(args[0], FakeTensor)
            if args[0].fake_mode.in_kernel_invocation:
                return torch.device('meta')
            else:
                return args[0].fake_device
        unrecognized_types = [t for t in types if not issubclass(t, FakeTensor) and t is not torch.Tensor]
        if unrecognized_types:
            not_implemented_log.debug('FakeTensor unrecognized subclass(es): %s', unrecognized_types)
            return NotImplemented
        fake_mode = None
        for arg in pytree.arg_tree_leaves(*args, **kwargs):
            if isinstance(arg, FakeTensor):
                fake_mode = arg.fake_mode
                break
        assert fake_mode is not None
        maybe_cur_fake_mode = torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE)
        if maybe_cur_fake_mode:
            not_implemented_log.debug('FakeTensor mode already active: %s in %s', fake_mode, maybe_cur_fake_mode)
            return NotImplemented
        with fake_mode:
            return func(*args, **kwargs)

    @staticmethod
    def _find_common_device(func, flat_args) -> Tuple[torch.device, bool]:
        common_device = None
        has_scalar_only_inputs = False
        is_cpu_zero_dim = None

        def cpu_zero_dim(t):
            return t.device.type == 'cpu' and t.dim() == 0

        def merge_devices(t):
            nonlocal common_device
            nonlocal is_cpu_zero_dim
            if not isinstance(t, FakeTensor):
                return
            if common_device is None:
                common_device = t.device
                is_cpu_zero_dim = cpu_zero_dim(t)
                return
            t_is_cpu_zero_dim = cpu_zero_dim(t)
            if t.device == common_device:
                if is_cpu_zero_dim:
                    is_cpu_zero_dim = t_is_cpu_zero_dim
                return
            if t_is_cpu_zero_dim:
                return
            if is_cpu_zero_dim:
                common_device = t.device
                is_cpu_zero_dim = t_is_cpu_zero_dim
                return
            raise RuntimeError(f'Unhandled FakeTensor Device Propagation for {func}, found two different devices {common_device}, {t.device}')
        for arg in flat_args:
            merge_devices(arg)
        if should_allow_numbers_as_tensors(func) and common_device is None:
            has_scalar_only_inputs = True
            common_device = torch.device('cpu')
        assert common_device is not None, f'Could not find common device for {func}'
        return (common_device, has_scalar_only_inputs)

    def tolist(self):
        assert self.dim() == 1, 'NYI for higher dims'
        shape_env = self.fake_mode.shape_env
        out = []
        for _ in range(self.shape[0]):
            s = shape_env.create_unbacked_symint()
            torch._constrain_as_size(s, min=2)
            out.append(s)
        return out
    __torch_function__ = torch._C._disabled_torch_function_impl