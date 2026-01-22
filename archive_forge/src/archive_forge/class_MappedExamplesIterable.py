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
class MappedExamplesIterable(_BaseExamplesIterable):

    def __init__(self, ex_iterable: _BaseExamplesIterable, function: Callable, with_indices: bool=False, input_columns: Optional[List[str]]=None, batched: bool=False, batch_size: Optional[int]=1000, drop_last_batch: bool=False, remove_columns: Optional[List[str]]=None, fn_kwargs: Optional[dict]=None, formatting: Optional['FormattingConfig']=None, format_type='deprecated'):
        if format_type != 'deprecated':
            warning_msg = "'format_type' is deprecated and will be removed in the next major version of datasets. "
            help_message = "Please use 'formatting=FormattingConfig(format_type=format_type)' instead."
            warnings.warn(warning_msg + help_message, category=FutureWarning, stacklevel=2)
            formatting = FormattingConfig(format_type=format_type)
        super().__init__()
        self.ex_iterable = ex_iterable
        self.function = function
        self.batched = batched
        self.batch_size = batch_size
        self.drop_last_batch = drop_last_batch
        self.remove_columns = remove_columns
        self.with_indices = with_indices
        self.input_columns = input_columns
        self.fn_kwargs = fn_kwargs or {}
        self.formatting = formatting
        if self.formatting and self.formatting.format_type == 'arrow':
            self.iter_arrow = self._iter_arrow

    def __iter__(self):
        if self.formatting and self.formatting.format_type == 'arrow':
            yield from ArrowExamplesIterable(self._iter_arrow, {})
        else:
            yield from self._iter()

    def _iter(self):
        iterator = iter(self.ex_iterable)
        current_idx = 0
        if self.formatting:
            formatter = get_formatter(self.formatting.format_type)
            format_dict = formatter.recursive_tensorize if isinstance(formatter, TensorFormatter) else cast_to_python_objects
        else:
            format_dict = None
        if self.batched:
            for key, example in iterator:
                iterator_batch = iterator if self.batch_size is None or self.batch_size <= 0 else islice(iterator, self.batch_size - 1)
                key_examples_list = [(key, example)] + list(iterator_batch)
                keys, examples = zip(*key_examples_list)
                if self.drop_last_batch and self.batch_size is not None and (self.batch_size > 0) and (len(examples) < self.batch_size):
                    return
                batch = _examples_to_batch(examples)
                batch = format_dict(batch) if format_dict else batch
                inputs = batch
                function_args = [inputs] if self.input_columns is None else [inputs[col] for col in self.input_columns]
                if self.with_indices:
                    function_args.append([current_idx + i for i in range(len(key_examples_list))])
                transformed_batch = dict(batch)
                transformed_batch.update(self.function(*function_args, **self.fn_kwargs))
                if self.remove_columns:
                    for c in self.remove_columns:
                        del transformed_batch[c]
                if transformed_batch:
                    first_col = next(iter(transformed_batch))
                    bad_cols = [col for col in transformed_batch if len(transformed_batch[col]) != len(transformed_batch[first_col])]
                    if bad_cols:
                        raise ValueError(f'Column lengths mismatch: columns {bad_cols} have length {[len(transformed_batch[col]) for col in bad_cols]} while {first_col} has length {len(transformed_batch[first_col])}.')
                new_key = '_'.join((str(key) for key in keys))
                for example in _batch_to_examples(transformed_batch):
                    yield (new_key, example)
                    current_idx += 1
        else:
            for key, example in iterator:
                example = dict(example)
                example = format_dict(example) if format_dict else example
                inputs = example
                function_args = [inputs] if self.input_columns is None else [inputs[col] for col in self.input_columns]
                if self.with_indices:
                    function_args.append(current_idx)
                transformed_example = dict(example)
                transformed_example.update(self.function(*function_args, **self.fn_kwargs))
                if self.remove_columns:
                    for c in self.remove_columns:
                        del transformed_example[c]
                yield (key, transformed_example)
                current_idx += 1

    def _iter_arrow(self) -> Iterator[Tuple[Key, pa.Table]]:
        if self.ex_iterable.iter_arrow:
            iterator = _batch_arrow_tables(self.ex_iterable.iter_arrow(), batch_size=self.batch_size if self.batched else 1, drop_last_batch=self.drop_last_batch)
        else:
            iterator = _convert_to_arrow(self.ex_iterable, batch_size=self.batch_size if self.batched else 1, drop_last_batch=self.drop_last_batch)
        current_idx = 0
        for key, pa_table in iterator:
            function_args = [pa_table] if self.input_columns is None else [pa_table[col] for col in self.input_columns]
            if self.with_indices:
                if self.batched:
                    function_args.append([current_idx + i for i in range(len(pa_table))])
                else:
                    function_args.append(current_idx)
            output_table = self.function(*function_args, **self.fn_kwargs)
            if not isinstance(output_table, pa.Table):
                raise TypeError(f'Provided `function` which is applied to pyarrow tables returns a variable of type {type(output_table)}. Make sure provided `function` returns a a pyarrow table to update the dataset.')
            if self.remove_columns:
                for column in self.remove_columns:
                    if column in output_table.column_names:
                        output_table = output_table.remove_column(output_table.column_names.index(column))
            yield (key, output_table)
            current_idx += len(pa_table)

    def shuffle_data_sources(self, generator: np.random.Generator) -> 'MappedExamplesIterable':
        """Shuffle the wrapped examples iterable."""
        return MappedExamplesIterable(self.ex_iterable.shuffle_data_sources(generator), function=self.function, with_indices=self.with_indices, input_columns=self.input_columns, batched=self.batched, batch_size=self.batch_size, drop_last_batch=self.drop_last_batch, remove_columns=self.remove_columns, fn_kwargs=self.fn_kwargs, formatting=self.formatting)

    def shard_data_sources(self, worker_id: int, num_workers: int) -> 'MappedExamplesIterable':
        """Keep only the requested shard."""
        return MappedExamplesIterable(self.ex_iterable.shard_data_sources(worker_id, num_workers), function=self.function, with_indices=self.with_indices, input_columns=self.input_columns, batched=self.batched, batch_size=self.batch_size, drop_last_batch=self.drop_last_batch, remove_columns=self.remove_columns, fn_kwargs=self.fn_kwargs, formatting=self.formatting)

    @property
    def n_shards(self) -> int:
        return self.ex_iterable.n_shards