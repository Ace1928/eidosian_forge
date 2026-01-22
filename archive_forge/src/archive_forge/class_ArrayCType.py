from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import auto, Enum
from typing import List, Optional, Union
from torchgen.model import Argument, SelfArgument, TensorOptionsArguments
@dataclass(frozen=True)
class ArrayCType(CType):
    elem: 'CType'
    size: int

    def cpp_type(self, *, strip_ref: bool=False) -> str:
        return f'::std::array<{self.elem.cpp_type()},{self.size}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'::std::array<{self.elem.cpp_type_registration_declarations()},{self.size}>'

    def remove_const_ref(self) -> 'CType':
        return ArrayCType(self.elem.remove_const_ref(), self.size)