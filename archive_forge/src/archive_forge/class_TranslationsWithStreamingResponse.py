from __future__ import annotations
from typing import Union, Mapping, cast
from typing_extensions import Literal
import httpx
from ... import _legacy_response
from ..._types import NOT_GIVEN, Body, Query, Headers, NotGiven, FileTypes
from ..._utils import (
from ..._compat import cached_property
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from ...types.audio import Translation, translation_create_params
from ..._base_client import (
class TranslationsWithStreamingResponse:

    def __init__(self, translations: Translations) -> None:
        self._translations = translations
        self.create = to_streamed_response_wrapper(translations.create)