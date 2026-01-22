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
class GitRefInfo:
    """
    Contains information about a git reference for a repo on the Hub.

    Attributes:
        name (`str`):
            Name of the reference (e.g. tag name or branch name).
        ref (`str`):
            Full git ref on the Hub (e.g. `"refs/heads/main"` or `"refs/tags/v1.0"`).
        target_commit (`str`):
            OID of the target commit for the ref (e.g. `"e7da7f221d5bf496a48136c0cd264e630fe9fcc8"`)
    """
    name: str
    ref: str
    target_commit: str