import copy
import itertools
import sys
import warnings
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from itertools import cycle, islice
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import numpy as np
import pyarrow as pa
from . import config
from .arrow_dataset import Dataset, DatasetInfoMixin
from .features import Features
from .features.features import FeatureType, _align_features, _check_if_features_can_be_aligned, cast_to_python_objects
from .filesystems import _reset_fsspec_lock
from .formatting import PythonFormatter, TensorFormatter, get_format_type_from_alias, get_formatter
from .info import DatasetInfo
from .splits import NamedSplit
from .table import cast_table_to_features, read_schema_from_file, table_cast
from .utils.logging import get_logger
from .utils.py_utils import Literal
from .utils.sharding import _merge_gen_kwargs, _number_of_shards_in_gen_kwargs, _shuffle_gen_kwargs, _split_gen_kwargs
def _iter_arrow(self) -> Iterator[Tuple[Key, pa.Table]]:
    schema = self.features.arrow_schema
    for key, pa_table in self.ex_iterable.iter_arrow():
        columns = set(pa_table.column_names)
        for column_name in self.features:
            if column_name not in columns:
                col = pa.NullArray.from_buffers(pa.null(), len(pa_table), [None])
                pa_table = pa_table.append_column(column_name, col)
        if pa_table.schema != schema:
            pa_table = cast_table_to_features(pa_table, self.features)
        yield (key, pa_table)