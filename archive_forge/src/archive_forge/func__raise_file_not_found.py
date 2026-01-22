import copy
import os
import re
import tempfile
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Union
from urllib.parse import quote, unquote
import fsspec
from requests import Response
from ._commit_api import CommitOperationCopy, CommitOperationDelete
from .constants import DEFAULT_REVISION, ENDPOINT, REPO_TYPE_MODEL, REPO_TYPES_MAPPING, REPO_TYPES_URL_PREFIXES
from .file_download import hf_hub_url
from .hf_api import HfApi, LastCommitInfo, RepoFile
from .utils import (
def _raise_file_not_found(path: str, err: Optional[Exception]) -> NoReturn:
    msg = path
    if isinstance(err, RepositoryNotFoundError):
        msg = f'{path} (repository not found)'
    elif isinstance(err, RevisionNotFoundError):
        msg = f'{path} (revision not found)'
    elif isinstance(err, HFValidationError):
        msg = f'{path} (invalid repository id)'
    raise FileNotFoundError(msg) from err