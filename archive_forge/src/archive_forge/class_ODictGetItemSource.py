import collections
import dataclasses
import enum
from typing import Any, Optional, Union
from torch._guards import ChainedSource, GuardSource, Source
from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr
@dataclasses.dataclass(frozen=True)
class ODictGetItemSource(ChainedSource):
    index: Any

    def __post_init__(self):
        assert self.base is not None

    def reconstruct(self, codegen):
        return [codegen._create_load_const(collections.OrderedDict.__getitem__), *reconstruct_getitem(self, codegen, index_is_slice=False), *create_call_function(2, True)]

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
        if isinstance(self.index, type):
            rep = f'__load_module("{self.index.__module__}").{self.index.__qualname__}'
            return f'___odict_getitem({self.base.name()}, {rep})'
        elif isinstance(self.index, Source):
            return f'___odict_getitem({self.base.name()}, {self.index.name()})'
        else:
            return f'___odict_getitem({self.base.name()}, {self.index!r})'