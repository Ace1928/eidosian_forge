from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
def topt_default_init(name: str) -> Optional[str]:
    topt_args = func.arguments.tensor_options
    if topt_args is None:
        return None
    a = getattr(topt_args, name)
    if a.default is None or a.default == 'None':
        return None
    return cpp.default_expr(a.default, a.type, symint=False)