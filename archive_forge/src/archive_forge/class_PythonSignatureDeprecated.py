from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
@dataclass(frozen=True)
class PythonSignatureDeprecated(PythonSignature):
    deprecated_schema: FunctionSchema
    deprecated_args_exprs: Tuple[str, ...]

    @property
    def deprecated(self) -> bool:
        return True

    def signature_str(self, *, skip_outputs: bool=False, symint: bool=True) -> str:
        return PythonSignature.signature_str(self, skip_outputs=skip_outputs, symint=symint) + '|deprecated'

    def signature_str_pyi(self, *, skip_outputs: bool=False) -> str:
        args = self.arguments(skip_outputs=skip_outputs)
        schema_formals: List[str] = [a.argument_str_pyi(method=self.method, deprecated=True) for a in args]
        positional_argc = len(self.input_args)
        if len(schema_formals) > positional_argc:
            schema_formals.insert(positional_argc, '*')
        returns_str = returns_str_pyi(self)
        return f'def {self.name}({', '.join(schema_formals)}) -> {returns_str}: ...'

    def signature_str_pyi_vararg(self, *, skip_outputs: bool=False) -> Optional[str]:
        return None