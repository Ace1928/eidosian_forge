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
class CommitInfo(str):
    """Data structure containing information about a newly created commit.

    Returned by any method that creates a commit on the Hub: [`create_commit`], [`upload_file`], [`upload_folder`],
    [`delete_file`], [`delete_folder`]. It inherits from `str` for backward compatibility but using methods specific
    to `str` is deprecated.

    Attributes:
        commit_url (`str`):
            Url where to find the commit.

        commit_message (`str`):
            The summary (first line) of the commit that has been created.

        commit_description (`str`):
            Description of the commit that has been created. Can be empty.

        oid (`str`):
            Commit hash id. Example: `"91c54ad1727ee830252e457677f467be0bfd8a57"`.

        pr_url (`str`, *optional*):
            Url to the PR that has been created, if any. Populated when `create_pr=True`
            is passed.

        pr_revision (`str`, *optional*):
            Revision of the PR that has been created, if any. Populated when
            `create_pr=True` is passed. Example: `"refs/pr/1"`.

        pr_num (`int`, *optional*):
            Number of the PR discussion that has been created, if any. Populated when
            `create_pr=True` is passed. Can be passed as `discussion_num` in
            [`get_discussion_details`]. Example: `1`.

        _url (`str`, *optional*):
            Legacy url for `str` compatibility. Can be the url to the uploaded file on the Hub (if returned by
            [`upload_file`]), to the uploaded folder on the Hub (if returned by [`upload_folder`]) or to the commit on
            the Hub (if returned by [`create_commit`]). Defaults to `commit_url`. It is deprecated to use this
            attribute. Please use `commit_url` instead.
    """
    commit_url: str
    commit_message: str
    commit_description: str
    oid: str
    pr_url: Optional[str] = None
    pr_revision: Optional[str] = field(init=False)
    pr_num: Optional[str] = field(init=False)
    _url: str = field(repr=False, default=None)

    def __new__(cls, *args, commit_url: str, _url: Optional[str]=None, **kwargs):
        return str.__new__(cls, _url or commit_url)

    def __post_init__(self):
        """Populate pr-related fields after initialization.

        See https://docs.python.org/3.10/library/dataclasses.html#post-init-processing.
        """
        if self.pr_url is not None:
            self.pr_revision = _parse_revision_from_pr_url(self.pr_url)
            self.pr_num = int(self.pr_revision.split('/')[-1])
        else:
            self.pr_revision = None
            self.pr_num = None