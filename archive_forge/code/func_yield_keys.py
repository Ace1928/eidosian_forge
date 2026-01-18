from typing import (
from langchain_core.stores import BaseStore
def yield_keys(self, *, prefix: Optional[str]=None) -> Union[Iterator[K], Iterator[str]]:
    """Get an iterator over keys that match the given prefix."""
    yield from self.store.yield_keys(prefix=prefix)