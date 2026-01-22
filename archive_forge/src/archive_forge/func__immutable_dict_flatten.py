from typing import Any, Dict, Iterable, List, Tuple
from ._compatibility import compatibility
from torch.utils._pytree import Context, register_pytree_node
def _immutable_dict_flatten(d: Dict[Any, Any]) -> Tuple[List[Any], Context]:
    return (list(d.values()), list(d.keys()))