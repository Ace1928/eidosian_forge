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
def _wrapped_lfs_upload(batch_action) -> None:
    try:
        operation = oid2addop[batch_action['oid']]
        lfs_upload(operation=operation, lfs_batch_action=batch_action, token=token)
    except Exception as exc:
        raise RuntimeError(f"Error while uploading '{operation.path_in_repo}' to the Hub.") from exc