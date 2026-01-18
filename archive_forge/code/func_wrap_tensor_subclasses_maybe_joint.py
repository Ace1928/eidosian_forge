from typing import Any, List, Optional, Tuple, Union
import torch.utils._pytree as pytree
from torch import Tensor
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from .schemas import SubclassCreationMeta, ViewAndMutationMeta
from .utils import strict_zip
def wrap_tensor_subclasses_maybe_joint(unwrapped_args, *, is_joint_structure: bool, meta: ViewAndMutationMeta) -> Union[Tuple[Any, ...], List[Any]]:
    if is_joint_structure:
        assert isinstance(unwrapped_args, tuple) and len(unwrapped_args) == 2
        assert isinstance(unwrapped_args[0], (tuple, list)) and isinstance(unwrapped_args[1], (tuple, list))
        primals, tangents = (unwrapped_args[0], unwrapped_args[1])
        wrapped_primals = wrap_tensor_subclasses(primals, subclass_metas=meta.subclass_inp_meta)
        wrapped_tangents = wrap_tensor_subclasses(tangents, subclass_metas=meta.subclass_tangent_meta)
        return (wrapped_primals, wrapped_tangents)
    else:
        wrapped_args = wrap_tensor_subclasses(unwrapped_args, subclass_metas=meta.subclass_inp_meta)
        return wrapped_args