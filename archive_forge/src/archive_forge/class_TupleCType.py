from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import auto, Enum
from typing import List, Optional, Union
from torchgen.model import Argument, SelfArgument, TensorOptionsArguments
@dataclass(frozen=True)
class TupleCType(CType):
    elems: List['CType']

    def cpp_type(self, *, strip_ref: bool=False) -> str:
        return f'::std::tuple<{','.join([e.cpp_type() for e in self.elems])}>'

    def cpp_type_registration_declarations(self) -> str:
        return f'::std::tuple<{','.join([e.cpp_type_registration_declarations() for e in self.elems])}>'

    def remove_const_ref(self) -> 'CType':
        return TupleCType([e.remove_const_ref() for e in self.elems])