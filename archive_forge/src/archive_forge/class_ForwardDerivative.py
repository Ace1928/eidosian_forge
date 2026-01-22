import re
from dataclasses import dataclass
from typing import cast, Dict, List, Match, Optional, Sequence, Set, Tuple
from torchgen import local
from torchgen.api import cpp
from torchgen.api.types import BaseCType, Binding, NamedCType, tensorListT
from torchgen.model import (
from torchgen.utils import IDENT_REGEX
@dataclass(frozen=True)
class ForwardDerivative:
    formula: str
    var_names: Tuple[str, ...]
    var_types: Tuple[Type, ...]
    required_inputs_fw_grad: Optional[Tuple[str, ...]]
    required_inputs_primal: Optional[Tuple[str, ...]]
    required_original_self_value: bool
    is_reusing_outplace_formula: bool