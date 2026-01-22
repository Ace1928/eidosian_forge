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
def _iter_pytorch(self):
    ex_iterable = self._prepare_ex_iterable_for_iteration()
    _reset_fsspec_lock()
    import torch.utils.data
    worker_info = torch.utils.data.get_worker_info()
    if self._is_main_process() and ex_iterable.n_shards < worker_info.num_workers:
        logger.warning(f'Too many dataloader workers: {worker_info.num_workers} (max is dataset.n_shards={ex_iterable.n_shards}). Stopping {worker_info.num_workers - ex_iterable.n_shards} dataloader workers.')
        logger.info(f"To parallelize data loading, we give each process some shards (or data sources) to process. Therefore it's unnecessary to have a number of workers greater than dataset.n_shards={ex_iterable.n_shards}. To enable more parallelism, please split the dataset in more files than {ex_iterable.n_shards}.")
    _log_prefix = f'node#{self._distributed.rank} ' if self._distributed else ''
    shards_indices = ex_iterable.split_shard_indices_by_worker(worker_info.id, worker_info.num_workers)
    if shards_indices:
        logger.debug(f"{_log_prefix}dataloader worker#{worker_info.id}, ': Starting to iterate over {len(shards_indices)}/{ex_iterable.n_shards} shards.")
        ex_iterable = ex_iterable.shard_data_sources(worker_id=worker_info.id, num_workers=worker_info.num_workers)
        if self._formatting:
            formatter = get_formatter(self._formatting.format_type, features=self.features)
            format_dict = formatter.recursive_tensorize if isinstance(formatter, TensorFormatter) else cast_to_python_objects
        else:
            format_dict = None
        if self._formatting and (ex_iterable.iter_arrow or self._formatting == 'arrow'):
            if ex_iterable.iter_arrow:
                iterator = _batch_arrow_tables(ex_iterable.iter_arrow(), batch_size=1)
            else:
                iterator = _convert_to_arrow(ex_iterable, batch_size=1)
            for key, pa_table in iterator:
                yield formatter.format_row(pa_table)
            return
        else:
            for key, example in ex_iterable:
                if self.features:
                    example = _apply_feature_types_on_example(example, self.features, token_per_repo_id=self._token_per_repo_id)
                yield (format_dict(example) if format_dict else example)
        logger.debug(f"{_log_prefix}dataloader worker#{worker_info.id}, ': Finished iterating over {len(shards_indices)}/{ex_iterable.n_shards} shards.")
    else:
        logger.debug(f"{_log_prefix}dataloader worker#{worker_info.id}, ': Stopping... Number of dataset shards < num_workers ({ex_iterable.n_shards}<{worker_info.num_workers}).")