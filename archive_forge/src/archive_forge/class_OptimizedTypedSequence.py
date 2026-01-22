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
class OptimizedTypedSequence(TypedSequence):

    def __init__(self, data, type: Optional[FeatureType]=None, try_type: Optional[FeatureType]=None, col: Optional[str]=None, optimized_int_type: Optional[FeatureType]=None):
        optimized_int_type_by_col = {'attention_mask': Value('int8'), 'special_tokens_mask': Value('int8'), 'input_ids': Value('int32'), 'token_type_ids': Value('int8')}
        if type is None and try_type is None:
            optimized_int_type = optimized_int_type_by_col.get(col, None)
        super().__init__(data, type=type, try_type=try_type, optimized_int_type=optimized_int_type)