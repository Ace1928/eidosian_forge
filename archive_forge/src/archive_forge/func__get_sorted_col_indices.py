import collections
import csv
import functools
import gzip
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import error_ops
from tensorflow.python.data.experimental.ops import parsing_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import map_op
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.data.util import convert
from tensorflow.python.data.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util.tf_export import tf_export
def _get_sorted_col_indices(select_columns, column_names):
    """Transforms select_columns argument into sorted column indices."""
    names_to_indices = {n: i for i, n in enumerate(column_names)}
    num_cols = len(column_names)
    results = []
    for v in select_columns:
        if isinstance(v, int):
            if v < 0 or v >= num_cols:
                raise ValueError(f'Column index {v} specified in `select_columns` should be > 0  and <= {num_cols}, which is the number of columns.')
            results.append(v)
        elif v not in names_to_indices:
            raise ValueError(f'Column {v} specified in `select_columns` must be of one of the columns: {names_to_indices.keys()}.')
        else:
            results.append(names_to_indices[v])
    results = sorted(set(results))
    if len(results) != len(select_columns):
        sorted_names = sorted(results)
        duplicate_columns = set([a for a, b in zip(sorted_names[:-1], sorted_names[1:]) if a == b])
        raise ValueError(f'The `select_columns` argument contains duplicate columns: {duplicate_columns}.')
    return results