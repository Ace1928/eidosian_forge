import collections
import dataclasses
import enum
from typing import Any, Optional, Union
from torch._guards import ChainedSource, GuardSource, Source
from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr
@dataclasses.dataclass(frozen=True)
class GlobalSource(Source):
    global_name: str

    def reconstruct(self, codegen):
        return [codegen.create_load_global(self.global_name, False, add=True)]

    def guard_source(self):
        return GuardSource.GLOBAL

    def name(self):
        return f'G[{repr(self.global_name)}]'