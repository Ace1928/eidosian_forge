from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
@dataclass(frozen=True)
class PythonSignatureGroup:
    signature: PythonSignature
    base: NativeFunction
    outplace: Optional[NativeFunction]

    @classmethod
    def from_pairs(cls, functional: PythonSignatureNativeFunctionPair, out: Optional[PythonSignatureNativeFunctionPair]) -> 'PythonSignatureGroup':
        if out is None:
            return PythonSignatureGroup(signature=functional.signature, base=functional.function, outplace=None)
        signature_kwargs = out.signature.__dict__.copy()
        signature_kwargs['tensor_options_args'] = functional.signature.tensor_options_args
        return PythonSignatureGroup(signature=type(out.signature)(**signature_kwargs), base=functional.function, outplace=out.function)