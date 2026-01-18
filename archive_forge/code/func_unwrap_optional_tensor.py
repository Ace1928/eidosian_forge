import textwrap
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
from torchgen.api.translate import translate
from torchgen.api.types import DispatcherSignature
from torchgen.context import method_with_native_function
from torchgen.model import (
from torchgen.utils import mapMaybe
def unwrap_optional_tensor(name: str, cur_level_var: str) -> List[str]:
    result = f'    optional<Tensor> {name}_value;\n    optional<int64_t> {name}_bdim;\n    if ({name}) {{\n        std::tie({name}_value, {name}_bdim) = unwrapTensorAtLevel({name}.value(), {cur_level_var});\n    }}'
    return textwrap.dedent(result).split('\n')