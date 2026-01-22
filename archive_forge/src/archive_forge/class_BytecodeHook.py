import dataclasses
import sys
import types
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Protocol, Union
from typing_extensions import TypeAlias
import torch
class BytecodeHook(Protocol):

    def __call__(self, code: types.CodeType, new_code: types.CodeType) -> Optional[types.CodeType]:
        ...