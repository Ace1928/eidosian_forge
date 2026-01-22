from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import auto, Enum
from typing import List, Optional, Union
from torchgen.model import Argument, SelfArgument, TensorOptionsArguments
@dataclass(frozen=True)
class ConstRefCType(CType):
    elem: 'CType'

    def cpp_type(self, *, strip_ref: bool=False) -> str:
        if strip_ref:
            return self.elem.cpp_type(strip_ref=strip_ref)
        return f'const {self.elem.cpp_type()} &'

    def cpp_type_registration_declarations(self) -> str:
        return f'const {self.elem.cpp_type_registration_declarations()} &'

    def remove_const_ref(self) -> 'CType':
        return self.elem.remove_const_ref()