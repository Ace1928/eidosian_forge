from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.ufunc as ufunc
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.api.ufunc import UfunctorBindings
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import OrderedSet
@dataclass(frozen=True)
class UfunctorSignature:
    g: NativeFunctionsGroup
    scalar_tensor_idx: Optional[int]
    name: str

    def arguments(self) -> UfunctorBindings:
        return ufunc.ufunctor_arguments(self.g, scalar_tensor_idx=self.scalar_tensor_idx, scalar_t=scalar_t)

    def fields(self) -> List[Binding]:
        return [b.rename(f'{b.name}_') for b in self.arguments().ctor]

    def returns_type(self) -> CType:
        return BaseCType(scalar_t)

    def decl_fields(self) -> str:
        return '\n'.join((f'{f.type} {f.name};' for f in self.fields()))

    def inline_defn_ctor(self) -> str:
        args_str = ', '.join((a.decl() for a in self.arguments().ctor))
        init_str = ', '.join((f'{a.name}_({a.name})' for a in self.arguments().ctor))
        return f'{self.name}({args_str}) : {init_str} {{}}'

    def decl_apply(self) -> str:
        args_str = ', '.join((a.decl() for a in self.arguments().apply))
        return f'{self.returns_type().cpp_type()} operator()({args_str}) const'