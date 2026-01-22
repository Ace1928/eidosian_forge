from __future__ import annotations
import builtins
import time
from typing import Dict
from ..testing import do_bench
from .jit import KernelInterface
def _pre_hook(args, reset_only=False):
    for i in self.reset_idx:
        args[i].zero_()
    if not reset_only:
        self.restore_copies = [args[i].clone() for i in self.restore_idx]