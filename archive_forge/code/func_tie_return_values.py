import itertools
from typing import Dict, List, Sequence, Union
from torchgen.api import cpp
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import Argument, NativeFunction, SchemaKind, TensorOptionsArguments
from torchgen.utils import FileManager
def tie_return_values(f: NativeFunction) -> str:
    if len(f.func.returns) == 1:
        return f'auto {f.func.returns[0].name or 'result'}'
    names = cpp.return_names(f)
    return f'std::tie({', '.join(names)})'