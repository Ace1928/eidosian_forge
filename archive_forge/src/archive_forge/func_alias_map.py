from typing import (Any, Callable, Dict, Generic, Iterator, List, Optional,
from .pyutils import get_named_object
def alias_map(self) -> Dict[K, List[K]]:
    ret: Dict[K, List[K]] = {}
    for alias, target in self._aliases.items():
        ret.setdefault(target, []).append(alias)
    return ret