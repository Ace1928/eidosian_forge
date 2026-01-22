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
class DeterministicAlgorithmsVariable(ContextWrappingVariable):
    """represents torch.{are_deterministic_algorithms_enabled,use_deterministic_algorithms}()"""
    _guards_singleton = Guard(GlobalStateSource(), GuardBuilder.DETERMINISTIC_ALGORITHMS)

    @staticmethod
    def create(tx, target_value, **kwargs):
        var = DeterministicAlgorithmsVariable(target_values=[target_value], initial_values=[torch.are_deterministic_algorithms_enabled()], **kwargs)
        var._call_func(tx, [target_value])
        var.set_cleanup_hook(tx)
        return var

    def __init__(self, target_values, initial_values=None, **kwargs):
        super().__init__(target_values=target_values, initial_values=initial_values, **kwargs)
        install_guard(self._guards_singleton)

    def enter(self, tx):
        return variables.ConstantVariable.create(None)

    def _call_func(self, tx, values):
        assert len(values) == 1
        value = values[0]
        (tx.output.create_node('call_function', torch._C._set_deterministic_algorithms, (value,), {}),)
        torch._C._set_deterministic_algorithms(value)

    def module_name(self):
        return 'torch'

    def fn_name(self):
        return 'use_deterministic_algorithms'