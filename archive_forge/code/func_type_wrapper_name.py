import itertools
from typing import Dict, List, Sequence, Union
from torchgen.api import cpp
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import Argument, NativeFunction, SchemaKind, TensorOptionsArguments
from torchgen.utils import FileManager
def type_wrapper_name(f: NativeFunction, key: str='Default') -> str:
    if f.func.name.overload_name:
        name = f'{cpp.name(f.func)}_{f.func.name.overload_name}'
    else:
        name = cpp.name(f.func)
    if key != 'Default':
        name = name + f'_{key}'
    return name