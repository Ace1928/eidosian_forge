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
def _list_access_requests(self, repo_id: str, status: Literal['accepted', 'rejected', 'pending'], repo_type: Optional[str]=None, token: Optional[str]=None) -> List[AccessRequest]:
    if repo_type not in REPO_TYPES:
        raise ValueError(f'Invalid repo type, must be one of {REPO_TYPES}')
    if repo_type is None:
        repo_type = REPO_TYPE_MODEL
    response = get_session().get(f'{ENDPOINT}/api/{repo_type}s/{repo_id}/user-access-request/{status}', headers=self._build_hf_headers(token=token))
    hf_raise_for_status(response)
    return [AccessRequest(username=request['user']['user'], fullname=request['user']['fullname'], email=request['user']['email'], status=request['status'], timestamp=parse_datetime(request['timestamp']), fields=request.get('fields')) for request in response.json()]