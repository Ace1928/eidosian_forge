import collections
import dataclasses
import enum
from typing import Any, Optional, Union
from torch._guards import ChainedSource, GuardSource, Source
from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr
@dataclasses.dataclass(frozen=True)
class DefaultsSource(ChainedSource):
    idx_key: Union[int, str]
    is_kw: bool = False
    field: str = dataclasses.field(init=False, repr=False, compare=False)
    _name: str = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self):
        assert self.base, 'Base must be a valid source in order to properly track and guard this Defaults to its origin.'
        if self.is_kw:
            assert isinstance(self.idx_key, str)
            object.__setattr__(self, 'field', '__kwdefaults__')
            object.__setattr__(self, '_name', f"{self.base.name()}.{self.field}['{self.idx_key}']")
        else:
            assert isinstance(self.idx_key, int)
            object.__setattr__(self, 'field', '__defaults__')
            object.__setattr__(self, '_name', f'{self.base.name()}.{self.field}[{self.idx_key}]')

    def reconstruct(self, codegen):
        instrs = self.base.reconstruct(codegen)
        instrs.extend(codegen.create_load_attrs(self.field))
        instrs.extend([codegen.create_load_const(self.idx_key), create_instruction('BINARY_SUBSCR')])
        return instrs

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
        return self._name