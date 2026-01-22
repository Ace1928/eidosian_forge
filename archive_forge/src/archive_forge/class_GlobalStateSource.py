import collections
import dataclasses
import enum
from typing import Any, Optional, Union
from torch._guards import ChainedSource, GuardSource, Source
from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr
@dataclasses.dataclass(frozen=True)
class GlobalStateSource(Source):

    def name(self):
        return ''

    def guard_source(self):
        return GuardSource.GLOBAL