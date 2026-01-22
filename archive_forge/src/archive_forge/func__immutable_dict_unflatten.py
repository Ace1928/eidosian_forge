from typing import Any, Dict, Iterable, List, Tuple
from ._compatibility import compatibility
from torch.utils._pytree import Context, register_pytree_node
def _immutable_dict_unflatten(values: Iterable[Any], context: Context) -> Dict[Any, Any]:
    return immutable_dict(dict(zip(context, values)))