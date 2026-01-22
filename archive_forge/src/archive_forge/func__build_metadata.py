import errno
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from . import config
from .features import Features, Image, Value
from .features.features import (
from .filesystems import is_remote_filesystem
from .info import DatasetInfo
from .keyhash import DuplicatedKeysError, KeyHasher
from .table import array_cast, cast_array_to_feature, embed_table_storage, table_cast
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.file_utils import hash_url_to_filename
from .utils.py_utils import asdict, first_non_null_value
@staticmethod
def _build_metadata(info: DatasetInfo, fingerprint: Optional[str]=None) -> Dict[str, str]:
    info_keys = ['features']
    info_as_dict = asdict(info)
    metadata = {}
    metadata['info'] = {key: info_as_dict[key] for key in info_keys}
    if fingerprint is not None:
        metadata['fingerprint'] = fingerprint
    return {'huggingface': json.dumps(metadata)}