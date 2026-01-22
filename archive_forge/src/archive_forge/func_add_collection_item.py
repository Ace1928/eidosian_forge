from __future__ import annotations
import inspect
import json
import re
import struct
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import wraps
from itertools import islice
from pathlib import Path
from typing import (
from urllib.parse import quote
import requests
from requests.exceptions import HTTPError
from tqdm.auto import tqdm as base_tqdm
from tqdm.contrib.concurrent import thread_map
from ._commit_api import (
from ._inference_endpoints import InferenceEndpoint, InferenceEndpointType
from ._multi_commits import (
from ._space_api import SpaceHardware, SpaceRuntime, SpaceStorage, SpaceVariable
from .community import (
from .constants import (
from .file_download import HfFileMetadata, get_hf_file_metadata, hf_hub_url
from .repocard_data import DatasetCardData, ModelCardData, SpaceCardData
from .utils import (  # noqa: F401 # imported for backward compatibility
from .utils import tqdm as hf_tqdm
from .utils._deprecation import _deprecate_arguments, _deprecate_method
from .utils._typing import CallableT
from .utils.endpoint_helpers import (
def add_collection_item(self, collection_slug: str, item_id: str, item_type: CollectionItemType_T, *, note: Optional[str]=None, exists_ok: bool=False, token: Optional[str]=None) -> Collection:
    """Add an item to a collection on the Hub.

        Args:
            collection_slug (`str`):
                Slug of the collection to update. Example: `"TheBloke/recent-models-64f9a55bb3115b4f513ec026"`.
            item_id (`str`):
                ID of the item to add to the collection. It can be the ID of a repo on the Hub (e.g. `"facebook/bart-large-mnli"`)
                or a paper id (e.g. `"2307.09288"`).
            item_type (`str`):
                Type of the item to add. Can be one of `"model"`, `"dataset"`, `"space"` or `"paper"`.
            note (`str`, *optional*):
                A note to attach to the item in the collection. The maximum size for a note is 500 characters.
            exists_ok (`bool`, *optional*):
                If `True`, do not raise an error if item already exists.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if not provided.

        Returns: [`Collection`]

        Raises:
            `HTTPError`:
                HTTP 403 if you only have read-only access to the repo. This can be the case if you don't have `write`
                or `admin` role in the organization the repo belongs to or if you passed a `read` token.
            `HTTPError`:
                HTTP 404 if the item you try to add to the collection does not exist on the Hub.
            `HTTPError`:
                HTTP 409 if the item you try to add to the collection is already in the collection (and exists_ok=False)

        Example:

        ```py
        >>> from huggingface_hub import add_collection_item
        >>> collection = add_collection_item(
        ...     collection_slug="davanstrien/climate-64f99dc2a5067f6b65531bab",
        ...     item_id="pierre-loic/climate-news-articles",
        ...     item_type="dataset"
        ... )
        >>> collection.items[-1].item_id
        "pierre-loic/climate-news-articles"
        # ^item got added to the collection on last position

        # Add item with a note
        >>> add_collection_item(
        ...     collection_slug="davanstrien/climate-64f99dc2a5067f6b65531bab",
        ...     item_id="datasets/climate_fever",
        ...     item_type="dataset"
        ...     note="This dataset adopts the FEVER methodology that consists of 1,535 real-world claims regarding climate-change collected on the internet."
        ... )
        (...)
        ```
        """
    payload: Dict[str, Any] = {'item': {'id': item_id, 'type': item_type}}
    if note is not None:
        payload['note'] = note
    r = get_session().post(f'{self.endpoint}/api/collections/{collection_slug}/items', headers=self._build_hf_headers(token=token), json=payload)
    try:
        hf_raise_for_status(r)
    except HTTPError as err:
        if exists_ok and err.response.status_code == 409:
            return self.get_collection(collection_slug, token=token)
        else:
            raise
    return Collection(**{**r.json(), 'endpoint': self.endpoint})