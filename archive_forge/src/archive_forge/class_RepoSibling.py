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
class RepoSibling:
    """
    Contains basic information about a repo file inside a repo on the Hub.

    <Tip>

    All attributes of this class are optional except `rfilename`. This is because only the file names are returned when
    listing repositories on the Hub (with [`list_models`], [`list_datasets`] or [`list_spaces`]). If you need more
    information like file size, blob id or lfs details, you must request them specifically from one repo at a time
    (using [`model_info`], [`dataset_info`] or [`space_info`]) as it adds more constraints on the backend server to
    retrieve these.

    </Tip>

    Attributes:
        rfilename (str):
            file name, relative to the repo root.
        size (`int`, *optional*):
            The file's size, in bytes. This attribute is defined when `files_metadata` argument of [`repo_info`] is set
            to `True`. It's `None` otherwise.
        blob_id (`str`, *optional*):
            The file's git OID. This attribute is defined when `files_metadata` argument of [`repo_info`] is set to
            `True`. It's `None` otherwise.
        lfs (`BlobLfsInfo`, *optional*):
            The file's LFS metadata. This attribute is defined when`files_metadata` argument of [`repo_info`] is set to
            `True` and the file is stored with Git LFS. It's `None` otherwise.
    """
    rfilename: str
    size: Optional[int] = None
    blob_id: Optional[str] = None
    lfs: Optional[BlobLfsInfo] = None