from __future__ import annotations
from .threads import (
from ..._compat import cached_property
from .assistants import (
from ..._resource import SyncAPIResource, AsyncAPIResource
from .vector_stores import (
from .threads.threads import Threads, AsyncThreads
from .vector_stores.vector_stores import VectorStores, AsyncVectorStores
@cached_property
def vector_stores(self) -> AsyncVectorStoresWithStreamingResponse:
    return AsyncVectorStoresWithStreamingResponse(self._beta.vector_stores)