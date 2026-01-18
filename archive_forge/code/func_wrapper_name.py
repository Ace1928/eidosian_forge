from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
from torchgen.api import cpp, dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
def wrapper_name(func: FunctionSchema) -> str:
    if func.name.overload_name:
        return f'{cpp.name(func)}_{func.name.overload_name}'
    else:
        return cpp.name(func)