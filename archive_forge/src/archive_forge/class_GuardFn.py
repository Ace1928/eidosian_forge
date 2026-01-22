import dataclasses
import sys
import types
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Protocol, Union
from typing_extensions import TypeAlias
import torch
class GuardFn(Protocol):
    closure_vars: Dict[str, object]
    args: List[str]
    code_parts: List[str]
    verbose_code_parts: List[str]
    global_scope: Dict[str, object]
    guard_fail_fn: Optional[Callable[[GuardFail], None]]

    def __call__(self, f_locals: Dict[str, object]) -> bool:
        ...