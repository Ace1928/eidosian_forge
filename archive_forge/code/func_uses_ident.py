import re
from dataclasses import dataclass
from typing import cast, Dict, List, Match, Optional, Sequence, Set, Tuple
from torchgen import local
from torchgen.api import cpp
from torchgen.api.types import BaseCType, Binding, NamedCType, tensorListT
from torchgen.model import (
from torchgen.utils import IDENT_REGEX
def uses_ident(info: Optional[DifferentiabilityInfo], ident: str) -> bool:
    if info is None:
        return False
    for derivative in info.derivatives:
        formula = derivative.formula
        if re.search(IDENT_REGEX.format(ident), formula):
            return True
    return False