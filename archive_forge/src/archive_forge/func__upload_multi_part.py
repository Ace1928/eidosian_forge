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
def _upload_multi_part(operation: 'CommitOperationAdd', header: Dict, chunk_size: int, upload_url: str) -> None:
    """
    Uploads file using HF multipart LFS transfer protocol.
    """
    sorted_parts_urls = _get_sorted_parts_urls(header=header, upload_info=operation.upload_info, chunk_size=chunk_size)
    use_hf_transfer = HF_HUB_ENABLE_HF_TRANSFER
    if HF_HUB_ENABLE_HF_TRANSFER and (not isinstance(operation.path_or_fileobj, str)) and (not isinstance(operation.path_or_fileobj, Path)):
        warnings.warn('hf_transfer is enabled but does not support uploading from bytes or BinaryIO, falling back to regular upload')
        use_hf_transfer = False
    response_headers = _upload_parts_hf_transfer(operation=operation, sorted_parts_urls=sorted_parts_urls, chunk_size=chunk_size) if use_hf_transfer else _upload_parts_iteratively(operation=operation, sorted_parts_urls=sorted_parts_urls, chunk_size=chunk_size)
    completion_res = get_session().post(upload_url, json=_get_completion_payload(response_headers, operation.upload_info.sha256.hex()), headers=LFS_HEADERS)
    hf_raise_for_status(completion_res)