from typing import Any, Iterator, List, Optional, Sequence, Tuple, cast
from langchain_core._api.deprecation import deprecated
from langchain_core.stores import BaseStore, ByteStore
class UpstashRedisByteStore(ByteStore):
    """
    BaseStore implementation using Upstash Redis
    as the underlying store to store raw bytes.
    """

    def __init__(self, *, client: Any=None, url: Optional[str]=None, token: Optional[str]=None, ttl: Optional[int]=None, namespace: Optional[str]=None) -> None:
        self.underlying_store = _UpstashRedisStore(client=client, url=url, token=token, ttl=ttl, namespace=namespace)

    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        """Get the values associated with the given keys."""
        return [value.encode('utf-8') if value is not None else None for value in self.underlying_store.mget(keys)]

    def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        """Set the given key-value pairs."""
        self.underlying_store.mset([(k, v.decode('utf-8')) if v is not None else None for k, v in key_value_pairs])

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys."""
        self.underlying_store.mdelete(keys)

    def yield_keys(self, *, prefix: Optional[str]=None) -> Iterator[str]:
        """Yield keys in the store."""
        yield from self.underlying_store.yield_keys(prefix=prefix)