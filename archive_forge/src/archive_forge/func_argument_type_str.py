from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
def argument_type_str(t: Type, *, simple_type: bool=False, symint: bool=True) -> str:
    if isinstance(t, BaseType):
        if t.name == BaseTy.Tensor:
            return 'Tensor'
        elif t.name == BaseTy.int:
            return 'int64_t'
        elif t.name == BaseTy.float:
            return 'double'
        elif t.name == BaseTy.str:
            return 'c10::string_view'
        elif t.name in [BaseTy.bool, BaseTy.QScheme, BaseTy.Scalar, BaseTy.ScalarType, BaseTy.Generator, BaseTy.Storage, BaseTy.Layout, BaseTy.Device, BaseTy.DeviceIndex, BaseTy.MemoryFormat, BaseTy.Dimname, BaseTy.Stream, BaseTy.ConstQuantizerPtr, BaseTy.SymInt]:
            return t.name.name
    elif isinstance(t, OptionalType):
        if str(t.elem) == 'Tensor':
            return 'Tensor?'
        elem = argument_type_str(t.elem, simple_type=simple_type, symint=symint)
        return f'{elem}?'
    elif isinstance(t, ListType):
        size = t.size if not simple_type else None
        if str(t.elem) == 'bool':
            assert t.size is not None
            return f'::std::array<bool,{t.size}>'
        elif str(t.elem) == 'int':
            return f'IntArrayRef[{size}]' if size is not None else 'IntArrayRef'
        elif str(t.elem) == 'SymInt':
            if symint:
                return f'SymIntArrayRef[{size}]' if size is not None else 'SymIntArrayRef'
            else:
                return f'IntArrayRef[{size}]' if size is not None else 'IntArrayRef'
        elif str(t.elem) == 'Tensor':
            return f'TensorList[{size}]' if size is not None else 'TensorList'
        elif str(t.elem) == 'Scalar':
            return f'ScalarList[{size}]' if size is not None else 'ScalarList'
        elif str(t.elem) == 'Tensor?':
            if simple_type:
                return 'c10::List<c10::optional<Tensor>>'
            else:
                return 'const c10::List<c10::optional<Tensor>> &'
        elif str(t.elem) == 'Dimname':
            return f'DimnameList[{size}]' if size is not None else 'DimnameList'
        elem = argument_type_str(t.elem, simple_type=simple_type, symint=symint)
        return f'ArrayRef<{elem}>'
    raise RuntimeError(f'unrecognized type {repr(t)}')