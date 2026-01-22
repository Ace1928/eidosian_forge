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
def _next_csv_row(filenames, num_cols, field_delim, use_quote_delim, header, file_io_fn):
    """Generator that yields rows of CSV file(s) in order."""
    for fn in filenames:
        with file_io_fn(fn) as f:
            rdr = csv.reader(f, delimiter=field_delim, quoting=csv.QUOTE_MINIMAL if use_quote_delim else csv.QUOTE_NONE)
            row_num = 1
            if header:
                next(rdr)
                row_num += 1
            for csv_row in rdr:
                if len(csv_row) != num_cols:
                    raise ValueError(f'Problem inferring types: CSV row {row_num} has {len(csv_row)} number of fields. Expected: {num_cols}.')
                row_num += 1
                yield csv_row