from dataclasses import dataclass
from typing import Dict
from torchgen.model import BaseTy, ScalarType
from .types_base import (
@dataclass(frozen=True)
class VectorizedCType(CType):
    elem: BaseCType

    def cpp_type(self, *, strip_ref: bool=False) -> str:
        return f'at::vec::Vectorized<{self.elem.cpp_type()}>'

    def cpp_type_registration_declarations(self) -> str:
        raise NotImplementedError

    def remove_const_ref(self) -> 'CType':
        return self