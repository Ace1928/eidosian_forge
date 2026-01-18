from __future__ import annotations
from .speech import (
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from .translations import (
from .transcriptions import (
@cached_property
def translations(self) -> AsyncTranslationsWithStreamingResponse:
    return AsyncTranslationsWithStreamingResponse(self._audio.translations)