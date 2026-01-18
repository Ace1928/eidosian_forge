from dataclasses import dataclass
from typing import List, Optional
import torchgen.api.types as api_types
from torchgen.api import cpp, structured
from torchgen.api.types import (
from torchgen.model import (
def ufunctor_ctor_type(t: Type, *, binds: ArgName, scalar_t: BaseCppType) -> NamedCType:
    r = cpp.valuetype_type(t, binds=binds, symint=False)
    if r is not None:
        return r
    if t == BaseType(BaseTy.Scalar):
        return NamedCType(binds, BaseCType(opmath_type(scalar_t)))
    elif t == BaseType(BaseTy.Tensor):
        return NamedCType(binds, BaseCType(opmath_type(scalar_t)))
    else:
        raise AssertionError(f'unrecognized type {repr(t)}')