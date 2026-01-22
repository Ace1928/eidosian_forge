import inspect
import io
import os
import re
import warnings
from contextlib import AbstractContextManager
from dataclasses import dataclass
from math import ceil
from os.path import getsize
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional, Tuple, TypedDict
from urllib.parse import unquote
from huggingface_hub.constants import ENDPOINT, HF_HUB_ENABLE_HF_TRANSFER, REPO_TYPES_URL_PREFIXES
from huggingface_hub.utils import get_session
from .utils import (
from .utils.sha import sha256, sha_fileobj
def _upload_single_part(operation: 'CommitOperationAdd', upload_url: str) -> None:
    """
    Uploads `fileobj` as a single PUT HTTP request (basic LFS transfer protocol)

    Args:
        upload_url (`str`):
            The URL to PUT the file to.
        fileobj:
            The file-like object holding the data to upload.

    Returns: `requests.Response`

    Raises: `requests.HTTPError` if the upload resulted in an error
    """
    with operation.as_file(with_tqdm=True) as fileobj:
        response = http_backoff('PUT', upload_url, data=fileobj, retry_on_status_codes=(500, 502, 503, 504))
        hf_raise_for_status(response)