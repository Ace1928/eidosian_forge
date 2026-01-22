import dataclasses
import inspect
from typing import Callable, Dict, List, Optional
import torch._C
from torch._guards import Guard
from .. import variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..device_interface import get_interface_for_device
from ..exc import unimplemented, Unsupported
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GlobalStateSource
from .base import VariableTracker
from .functions import (
class AutocastModeVariable(ContextWrappingVariable):

    @staticmethod
    def create(func, args, kwargs):
        assert func in [torch.amp.autocast_mode.autocast, torch.cuda.amp.autocast, torch.cpu.amp.autocast]
        bound_args = inspect.signature(func).bind(*args, **kwargs)
        bound_args.apply_defaults()
        target_values = []
        kwargs.clear()
        for key in ['device_type', 'dtype', 'enabled', 'cache_enabled']:
            if key == 'device_type' and func in [torch.cuda.amp.autocast, torch.cpu.amp.autocast]:
                arg = 'cuda' if func is torch.cuda.amp.autocast else 'cpu'
            else:
                arg = bound_args.arguments[key]
            if isinstance(arg, VariableTracker):
                target_values.append(arg.as_python_constant())
            else:
                target_values.append(arg)
        var = AutocastModeVariable(target_values, initial_values=None, **kwargs)
        return var

    def __init__(self, target_values, initial_values=None, **kwargs):
        super().__init__(target_values=target_values, initial_values=initial_values, **kwargs)
        self.target_values = target_values

    def exit(self, tx, *args):
        self.state.cleanup_assert()
        tx.output.create_node('call_function', torch.amp._exit_autocast, (self.state.proxy,), {})

    def enter(self, tx):
        ctx = torch.amp._enter_autocast(*self.target_values)
        self.set_cleanup_hook(tx, lambda: torch.amp._exit_autocast(ctx))
        self.state.proxy = tx.output.create_node('call_function', torch.amp._enter_autocast, (*self.target_values,), {})

    def module_name(self):
        return 'torch.amp.autocast_mode'

    def fn_name(self):
        return 'autocast'