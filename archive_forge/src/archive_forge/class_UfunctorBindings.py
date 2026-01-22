from dataclasses import dataclass
from typing import List, Optional
import torchgen.api.types as api_types
from torchgen.api import cpp, structured
from torchgen.api.types import (
from torchgen.model import (
@dataclass(frozen=True)
class UfunctorBindings:
    ctor: List[Binding]
    apply: List[Binding]