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
def _build_writer(self, inferred_schema: pa.Schema):
    schema = self.schema
    inferred_features = Features.from_arrow_schema(inferred_schema)
    if self._features is not None:
        if self.update_features:
            fields = {field.name: field for field in self._features.type}
            for inferred_field in inferred_features.type:
                name = inferred_field.name
                if name in fields:
                    if inferred_field == fields[name]:
                        inferred_features[name] = self._features[name]
            self._features = inferred_features
            schema: pa.Schema = inferred_schema
    else:
        self._features = inferred_features
        schema: pa.Schema = inferred_features.arrow_schema
    if self.disable_nullable:
        schema = pa.schema((pa.field(field.name, field.type, nullable=False) for field in schema))
    if self.with_metadata:
        schema = schema.with_metadata(self._build_metadata(DatasetInfo(features=self._features), self.fingerprint))
    else:
        schema = schema.with_metadata({})
    self._schema = schema
    self.pa_writer = self._WRITER_CLASS(self.stream, schema)