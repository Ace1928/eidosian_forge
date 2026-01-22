import collections
import dataclasses
import functools
import inspect
import sys
from typing import Any, Dict, List, Optional
import torch
import torch.fx
from .. import variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..eval_frame import skip_code
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard, make_dupe_guard
from ..source import AttrSource, GetItemSource, GlobalWeakRefSource
from ..utils import global_key_name, istensor, istype, iter_contains
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable
from .tensor import TensorVariable
@dataclasses.dataclass
class SetElement:
    vt: VariableTracker
    underlying_value: Any

    def __hash__(self) -> int:
        return hash(self.underlying_value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SetVariable.SetElement):
            return False
        if isinstance(self.vt, variables.TensorVariable):
            return self.underlying_value is other.underlying_value
        else:
            return self.underlying_value == other.underlying_value