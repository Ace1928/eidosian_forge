from dataclasses import dataclass
from typing import List, Optional
import torchgen.api.types as api_types
from torchgen.api import cpp, structured
from torchgen.api.types import (
from torchgen.model import (
def ufunctor_apply_type(t: Type, *, binds: ArgName, scalar_t: BaseCppType) -> NamedCType:
    if t == BaseType(BaseTy.Tensor):
        return NamedCType(binds, BaseCType(scalar_t))
    else:
        raise AssertionError(f'unrecognized type {repr(t)}')