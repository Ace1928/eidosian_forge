import re
from dataclasses import dataclass
from typing import cast, Dict, List, Match, Optional, Sequence, Set, Tuple
from torchgen import local
from torchgen.api import cpp
from torchgen.api.types import BaseCType, Binding, NamedCType, tensorListT
from torchgen.model import (
from torchgen.utils import IDENT_REGEX
@dataclass(frozen=True)
class Derivative:
    formula: str
    original_formula: str
    var_names: Tuple[str, ...]
    saved_inputs: Tuple[SavedAttribute, ...]
    saved_outputs: Tuple[SavedAttribute, ...]
    named_gradients: Set[str]