import textwrap
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from torchgen.api.translate import translate
from torchgen.api.types import DispatcherSignature
from torchgen.context import method_with_native_function
from torchgen.model import (
from torchgen.utils import mapMaybe
def accepts_at_least_one_tensor_input(schema: FunctionSchema) -> bool:
    return any((a.type.is_tensor_like() for a in schema.arguments.flat_all))