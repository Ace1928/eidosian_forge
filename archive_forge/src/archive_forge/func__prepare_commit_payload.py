import base64
import io
import os
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, BinaryIO, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union
from tqdm.contrib.concurrent import thread_map
from huggingface_hub import get_session
from .constants import ENDPOINT, HF_HUB_ENABLE_HF_TRANSFER
from .file_download import hf_hub_url
from .lfs import UploadInfo, lfs_upload, post_lfs_batch_info
from .utils import (
from .utils import tqdm as hf_tqdm
def _prepare_commit_payload(operations: Iterable[CommitOperation], files_to_copy: Dict[Tuple[str, Optional[str]], Union['RepoFile', bytes]], commit_message: str, commit_description: Optional[str]=None, parent_commit: Optional[str]=None) -> Iterable[Dict[str, Any]]:
    """
    Builds the payload to POST to the `/commit` API of the Hub.

    Payload is returned as an iterator so that it can be streamed as a ndjson in the
    POST request.

    For more information, see:
        - https://github.com/huggingface/huggingface_hub/issues/1085#issuecomment-1265208073
        - http://ndjson.org/
    """
    commit_description = commit_description if commit_description is not None else ''
    header_value = {'summary': commit_message, 'description': commit_description}
    if parent_commit is not None:
        header_value['parentCommit'] = parent_commit
    yield {'key': 'header', 'value': header_value}
    nb_ignored_files = 0
    for operation in operations:
        if isinstance(operation, CommitOperationAdd) and operation._should_ignore:
            logger.debug(f"Skipping file '{operation.path_in_repo}' in commit (ignored by gitignore file).")
            nb_ignored_files += 1
            continue
        if isinstance(operation, CommitOperationAdd) and operation._upload_mode == 'regular':
            yield {'key': 'file', 'value': {'content': operation.b64content().decode(), 'path': operation.path_in_repo, 'encoding': 'base64'}}
        elif isinstance(operation, CommitOperationAdd) and operation._upload_mode == 'lfs':
            yield {'key': 'lfsFile', 'value': {'path': operation.path_in_repo, 'algo': 'sha256', 'oid': operation.upload_info.sha256.hex(), 'size': operation.upload_info.size}}
        elif isinstance(operation, CommitOperationDelete):
            yield {'key': 'deletedFolder' if operation.is_folder else 'deletedFile', 'value': {'path': operation.path_in_repo}}
        elif isinstance(operation, CommitOperationCopy):
            file_to_copy = files_to_copy[operation.src_path_in_repo, operation.src_revision]
            if isinstance(file_to_copy, bytes):
                yield {'key': 'file', 'value': {'content': base64.b64encode(file_to_copy).decode(), 'path': operation.path_in_repo, 'encoding': 'base64'}}
            elif file_to_copy.lfs:
                yield {'key': 'lfsFile', 'value': {'path': operation.path_in_repo, 'algo': 'sha256', 'oid': file_to_copy.lfs.sha256}}
            else:
                raise ValueError('Malformed files_to_copy (should be raw file content as bytes or RepoFile objects with LFS info.')
        else:
            raise ValueError(f'Unknown operation to commit. Operation: {operation}. Upload mode: {getattr(operation, '_upload_mode', None)}')
    if nb_ignored_files > 0:
        logger.info(f'Skipped {nb_ignored_files} file(s) in commit (ignored by gitignore file).')