from typing import Any, Dict, Iterable, List, Tuple
from ._compatibility import compatibility
from torch.utils._pytree import Context, register_pytree_node
def _no_mutation(self, *args, **kwargs):
    raise NotImplementedError(f"'{type(self).__name__}' object does not support mutation. {_help_mutation}")