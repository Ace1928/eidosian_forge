from __future__ import annotations
import time
from typing import TYPE_CHECKING
import anyio
class SyncAPIResource:
    _client: OpenAI

    def __init__(self, client: OpenAI) -> None:
        self._client = client
        self._get = client.get
        self._post = client.post
        self._patch = client.patch
        self._put = client.put
        self._delete = client.delete
        self._get_api_list = client.get_api_list

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)