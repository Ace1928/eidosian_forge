from dataclasses import dataclass
from typing import List, Optional
import torchgen.api.types as api_types
from torchgen.api import cpp, structured
from torchgen.api.types import (
from torchgen.model import (
def ufunctor_arguments(g: NativeFunctionsGroup, *, scalar_tensor_idx: Optional[int], scalar_t: BaseCppType) -> UfunctorBindings:
    ctor = []
    apply = []
    for a in g.functional.func.arguments.flat_non_out:
        if a.type.is_tensor_like():
            if scalar_tensor_idx == 0:
                ctor.append(ufunctor_ctor_argument(a, scalar_t=scalar_t))
                scalar_tensor_idx = None
            else:
                if scalar_tensor_idx is not None:
                    scalar_tensor_idx -= 1
                apply.append(ufunctor_apply_argument(a, scalar_t=scalar_t))
        else:
            ctor.append(ufunctor_ctor_argument(a, scalar_t=scalar_t))
    assert scalar_tensor_idx is None
    return UfunctorBindings(ctor=ctor, apply=apply)