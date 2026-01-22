from typing import Any, Dict, Optional, Union, Iterable, List, Type, TYPE_CHECKING
from lazyops.utils.lazy import get_keydb_session
from .base import BaseStatefulBackend, SchemaType, logger
def _fetch_hset_keys(self, decode: Optional[bool]=True) -> List[str]:
    """
        This is a utility func for hset
        """
    keys: List[Union[str, bytes]] = self.cache.hkeys(self.base_key)
    if decode:
        return [key.decode() if isinstance(key, bytes) else key for key in keys]
    return keys