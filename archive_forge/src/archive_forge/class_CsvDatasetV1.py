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
@tf_export(v1=['data.experimental.CsvDataset'])
class CsvDatasetV1(dataset_ops.DatasetV1Adapter):
    """A Dataset comprising lines from one or more CSV files."""

    @functools.wraps(CsvDatasetV2.__init__, ('__module__', '__name__'))
    def __init__(self, filenames, record_defaults, compression_type=None, buffer_size=None, header=False, field_delim=',', use_quote_delim=True, na_value='', select_cols=None):
        """Creates a `CsvDataset` by reading and decoding CSV files.

    The elements of this dataset correspond to records from the file(s).
    RFC 4180 format is expected for CSV files
    (https://tools.ietf.org/html/rfc4180)
    Note that we allow leading and trailing spaces with int or float field.


    For example, suppose we have a file 'my_file0.csv' with four CSV columns of
    different data types:
    ```
    abcdefg,4.28E10,5.55E6,12
    hijklmn,-5.3E14,,2
    ```

    We can construct a CsvDataset from it as follows:

    ```python
     dataset = tf.data.experimental.CsvDataset(
        "my_file*.csv",
        [tf.float32,  # Required field, use dtype or empty tensor
         tf.constant([0.0], dtype=tf.float32),  # Optional field, default to 0.0
         tf.int32,  # Required field, use dtype or empty tensor
         ],
        select_cols=[1,2,3]  # Only parse last three columns
    )
    ```

    The expected output of its iterations is:

    ```python
    for element in dataset:
      print(element)

    >> (4.28e10, 5.55e6, 12)
    >> (-5.3e14, 0.0, 2)
    ```

    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      record_defaults: A list of default values for the CSV fields. Each item in
        the list is either a valid CSV `DType` (float32, float64, int32, int64,
        string), or a `Tensor` object with one of the above types. One per
        column of CSV data, with either a scalar `Tensor` default value for the
        column if it is optional, or `DType` or empty `Tensor` if required. If
        both this and `select_columns` are specified, these must have the same
        lengths, and `column_defaults` is assumed to be sorted in order of
        increasing column index. If both this and 'exclude_cols' are specified,
        the sum of lengths of record_defaults and exclude_cols should equal the
        total number of columns in the CSV file.
      compression_type: (Optional.) A `tf.string` scalar evaluating to one of
        `""` (no compression), `"ZLIB"`, or `"GZIP"`. Defaults to no
        compression.
      buffer_size: (Optional.) A `tf.int64` scalar denoting the number of bytes
        to buffer while reading files. Defaults to 4MB.
      header: (Optional.) A `tf.bool` scalar indicating whether the CSV file(s)
        have header line(s) that should be skipped when parsing. Defaults to
        `False`.
      field_delim: (Optional.) A `tf.string` scalar containing the delimiter
        character that separates fields in a record. Defaults to `","`.
      use_quote_delim: (Optional.) A `tf.bool` scalar. If `False`, treats double
        quotation marks as regular characters inside of string fields (ignoring
        RFC 4180, Section 2, Bullet 5). Defaults to `True`.
      na_value: (Optional.) A `tf.string` scalar indicating a value that will be
        treated as NA/NaN.
      select_cols: (Optional.) A sorted list of column indices to select from
        the input data. If specified, only this subset of columns will be
        parsed. Defaults to parsing all columns. At most one of `select_cols`
        and `exclude_cols` can be specified.
    """
        wrapped = CsvDatasetV2(filenames, record_defaults, compression_type, buffer_size, header, field_delim, use_quote_delim, na_value, select_cols)
        super(CsvDatasetV1, self).__init__(wrapped)