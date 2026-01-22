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
def _resolve_features(self):
    if self.features is not None:
        return self
    elif isinstance(self._ex_iterable, TypedExamplesIterable):
        features = self._ex_iterable.features
    else:
        features = _infer_features_from_batch(self.with_format(None)._head())
    info = self.info.copy()
    info.features = features
    return IterableDataset(ex_iterable=self._ex_iterable, info=info, split=self._split, formatting=self._formatting, shuffling=copy.deepcopy(self._shuffling), distributed=copy.deepcopy(self._distributed), token_per_repo_id=self._token_per_repo_id)