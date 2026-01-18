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
@validate_hf_hub_args
@_deprecate_arguments(version='0.24.0', deprecated_args=('organization', 'name'), custom_message='Use `repo_id` instead.')
def update_repo_visibility(self, repo_id: str, private: bool=False, *, token: Optional[str]=None, organization: Optional[str]=None, repo_type: Optional[str]=None, name: Optional[str]=None) -> Dict[str, bool]:
    """Update the visibility setting of a repository.

        Args:
            repo_id (`str`, *optional*):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            private (`bool`, *optional*, defaults to `False`):
                Whether the model repo should be private.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.

        Returns:
            The HTTP response in json.

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
    if repo_type not in REPO_TYPES:
        raise ValueError('Invalid repo type')
    organization, name = repo_id.split('/') if '/' in repo_id else (None, repo_id)
    if organization is None:
        namespace = self.whoami(token)['name']
    else:
        namespace = organization
    if repo_type is None:
        repo_type = REPO_TYPE_MODEL
    r = get_session().put(url=f'{self.endpoint}/api/{repo_type}s/{namespace}/{name}/settings', headers=self._build_hf_headers(token=token, is_write_action=True), json={'private': private})
    hf_raise_for_status(r)
    return r.json()