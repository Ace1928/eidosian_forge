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
@dataclass
class GitRefs:
    """
    Contains information about all git references for a repo on the Hub.

    Object is returned by [`list_repo_refs`].

    Attributes:
        branches (`List[GitRefInfo]`):
            A list of [`GitRefInfo`] containing information about branches on the repo.
        converts (`List[GitRefInfo]`):
            A list of [`GitRefInfo`] containing information about "convert" refs on the repo.
            Converts are refs used (internally) to push preprocessed data in Dataset repos.
        tags (`List[GitRefInfo]`):
            A list of [`GitRefInfo`] containing information about tags on the repo.
        pull_requests (`List[GitRefInfo]`, *optional*):
            A list of [`GitRefInfo`] containing information about pull requests on the repo.
            Only returned if `include_prs=True` is set.
    """
    branches: List[GitRefInfo]
    converts: List[GitRefInfo]
    tags: List[GitRefInfo]
    pull_requests: Optional[List[GitRefInfo]] = None