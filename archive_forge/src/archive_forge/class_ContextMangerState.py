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
@dataclasses.dataclass
class ContextMangerState:
    """
    Mutating `self` in VariableTracker is not allowed because we copy
    them.  This is a mutable container pointed to by context managers
    that won't get copied, so it is safe to mutate.
    """
    cleanup_fn: Optional[Callable] = None
    proxy: Optional[torch.fx.Proxy] = None

    def cleanup(self):
        if self.cleanup_fn is not None:
            self.cleanup_fn()
            self.cleanup_fn = None

    def cleanup_assert(self):
        assert self.cleanup_fn, 'multiple exits?'
        self.cleanup()