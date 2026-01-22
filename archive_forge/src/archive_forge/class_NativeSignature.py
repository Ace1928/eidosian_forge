from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Set, Tuple, Union
from torchgen.model import (
from .types_base import Binding, CType, Expr
from torchgen.api import (
@dataclass(frozen=True)
class NativeSignature:
    func: FunctionSchema
    symint: bool
    prefix: str = ''

    def name(self) -> str:
        return self.prefix + native.name(self.func)

    def decl(self, name: Optional[str]=None) -> str:
        args_str = ', '.join((a.decl() for a in self.arguments()))
        if name is None:
            name = self.name()
        return f'{native.returns_type(self.func.returns, symint=self.symint).cpp_type()} {name}({args_str})'

    def defn(self, name: Optional[str]=None) -> str:
        args_str = ', '.join((a.defn() for a in self.arguments()))
        if name is None:
            name = self.name()
        return f'{native.returns_type(self.func.returns, symint=self.symint).cpp_type()} {name}({args_str})'

    def ptr_type(self) -> str:
        args_str = ', '.join((a.defn() for a in self.arguments()))
        return f'{native.returns_type(self.func.returns, symint=self.symint).cpp_type()} (*)({args_str})'

    def arguments(self) -> List[Binding]:
        return native.arguments(self.func, symint=self.symint)

    def returns_type(self) -> CType:
        return native.returns_type(self.func.returns, symint=self.symint)

    def dispatcher_exprs(self) -> List[Expr]:
        return translate.translate(self.arguments(), dispatcher.arguments(self.func), method=False)