from typing import Any, List, Optional, Tuple, Union
import torch.utils._pytree as pytree
from torch import Tensor
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from .schemas import SubclassCreationMeta, ViewAndMutationMeta
from .utils import strict_zip
def wrap_tensor_subclasses(unwrapped_args: Union[Tuple[Any, ...], List[Any]], *, subclass_metas: List[Union[int, SubclassCreationMeta]], num_fw_outs_saved_for_bw: Optional[int]=None, is_runtime: bool=False) -> Tuple[Any, ...]:
    wrapped_args = []
    num_args_tallied = 0
    for subclass_meta in subclass_metas:
        if isinstance(subclass_meta, int):
            wrapped_args.append(unwrapped_args[subclass_meta])
            num_args_tallied += 1
        else:
            assert isinstance(subclass_meta, SubclassCreationMeta)
            wrapped_args.append(subclass_meta.creation_fn(unwrapped_args, is_runtime=is_runtime))
            num_args_tallied += subclass_meta.arg_count
    if num_fw_outs_saved_for_bw is not None:
        assert len(unwrapped_args) == num_args_tallied + num_fw_outs_saved_for_bw
        activations = unwrapped_args[num_args_tallied:]
        if isinstance(wrapped_args, tuple) and isinstance(activations, tuple):
            return wrapped_args + activations
        return tuple(list(wrapped_args) + list(activations))
    else:
        assert len(unwrapped_args) == num_args_tallied
        return tuple(wrapped_args)