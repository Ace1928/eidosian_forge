import importlib
import logging
import os
import pathlib
import random
import sys
import threading
import time
import urllib.parse
from collections import deque
from types import ModuleType
from typing import (
import numpy as np
import ray
from ray._private.utils import _get_pyarrow_version
from ray.data._internal.arrow_ops.transform_pyarrow import unify_schemas
from ray.data.context import WARN_PREFIX, DataContext
def unify_block_metadata_schema(metadata: List['BlockMetadata']) -> Optional[Union[type, 'pyarrow.lib.Schema']]:
    """For the input list of BlockMetadata, return a unified schema of the
    corresponding blocks. If the metadata have no valid schema, returns None.
    """
    schemas_to_unify = []
    for m in metadata:
        if m.schema is not None and (m.num_rows is None or m.num_rows > 0):
            schemas_to_unify.append(m.schema)
    if schemas_to_unify:
        try:
            import pyarrow as pa
        except ImportError:
            pa = None
        if pa is not None and any((isinstance(s, pa.Schema) for s in schemas_to_unify)):
            return unify_schemas(schemas_to_unify)
        return schemas_to_unify[0]
    return None