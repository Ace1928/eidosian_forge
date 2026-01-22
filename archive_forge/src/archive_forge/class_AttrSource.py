import collections
import dataclasses
import enum
from typing import Any, Optional, Union
from torch._guards import ChainedSource, GuardSource, Source
from . import utils
from .bytecode_transformation import create_call_function, create_instruction
from .utils import enum_repr
@dataclasses.dataclass(frozen=True)
class AttrSource(ChainedSource):
    member: str

    def __post_init__(self):
        assert self.base, "Can't construct an AttrSource without a valid base source"
        if '.' in self.member:
            member_parts = self.member.split('.')
            object.__setattr__(self, 'base', AttrSource(self.base, '.'.join(member_parts[:-1])))
            object.__setattr__(self, 'member', member_parts[-1])

    def reconstruct(self, codegen):
        return self.base.reconstruct(codegen) + codegen.create_load_attrs(self.member)

    def guard_source(self):
        return self.base.guard_source()

    def name(self):
        if not self.member.isidentifier():
            return f'getattr({self.base.name()}, {self.member!r})'
        return f'{self.base.name()}.{self.member}'