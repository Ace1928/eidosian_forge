import math
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_checkpoint_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
def _load_and_remap_matrix(ckpt_path, old_tensor_name, new_row_vocab_offset, num_rows_to_load, new_col_vocab_size, initializer, old_row_vocab_size=-1, old_row_vocab_file=None, new_row_vocab_file=None, old_col_vocab_file=None, new_col_vocab_file=None, num_row_oov_buckets=0, num_col_oov_buckets=0, max_rows_in_memory=-1):
    """Loads a 2-D (matrix) `Tensor` from checkpoint.

  Generates 1D-remappings for rows and columns using the
  `GenerateVocabRemapping` op, and initializes any anticipated values with the
  provided initializer. Then, uses the `LoadAndRemapMatrix` op to create a
  matrix that loads existing values from the checkpoint, while filling out
  "missing" values with the newly initialized values. See
  contrib/framework/ops/checkpoint_ops.cc for more information on the wrapped
  functionality (LoadAndRemapMatrix). This wrapper can be used to perform only
  row remapping or only col remapping. If only row remapping is desired,
  {new,old}_col_vocab_file should be `None`, and vice versa for column
  remapping.

  NOTE: This only supports div-partitioning the vocabulary on the 1st dimension
  (row axis) via `new_row_vocab_offset`.

  Args:
    ckpt_path: Path to the TensorFlow checkpoint (version 2, `TensorBundle`)
      from which the old matrix `Tensor` will be loaded.
    old_tensor_name: Name of the 2-D `Tensor` to load from checkpoint.
    new_row_vocab_offset: A 0-indexed integer representing what line to
      start reading at in the new row vocabulary. Used for partitioned
      variables.
    num_rows_to_load: Number of rows to load for the new vocabulary (note: to
      support variable partitioning and partial loading, this does not need to
      be the same as the number of entries in `new_row_vocab_file`).
    new_col_vocab_size: Number of columns to load - should be the same as the
      number of entries in `new_col_vocab_file`, since we don't support
      partitioning along the column axis.
    initializer: Callable initializer function that accepts a 1-D tensor as the
      arg to specify the shape of the returned tensor. Used to initialize
      missing values.
    old_row_vocab_size: The number of entries to consider in the old vocabulary.
      With the default value of -1, the entire old row vocabulary file will be
      used.  Otherwise, only the first `old_row_vocab_size` entries will be
      considered for remapping.Must be smaller than the length of
      `old_row_vocab_file`.  NOTE: we do not provide an equivalent
      `old_col_vocab_size` for classes.
    old_row_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the old row vocabulary file. Can be None, which represents no
      remapping on the row axis.
    new_row_vocab_file: A scalar `Tensor` of type `string` containing the path
      to the new row vocabulary file. Can be None, which represents no remapping
      on the row axis - in which case, `new_row_vocab_offset` and
      `num_rows_to_load` work under the assumption that the new row vocab is the
      same as the old row vocab.
    old_col_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the old column vocabulary file. Can be None, which represents no
      remapping on the column axis.
    new_col_vocab_file: A scalar `Tensor` of type `string` containing the path
      to the new column vocabulary file. Can be None, which represents no
      remapping on the column axis - in which case, `new_col_vocab_size` works
      under the assumption that the new col vocab is the same as the old col
      vocab.
    num_row_oov_buckets: `int` specifying the number of out-of-vocabulary rows
      to append. Must be >= 0.
    num_col_oov_buckets: `int` specifying the number of out-of-vocabulary
      columns to append. Must be >= 0.
    max_rows_in_memory: `int` specifying the maximum number of rows to load from
      the checkpoint at once. If less than or equal to 0, the entire matrix will
      be loaded into memory. Setting this arg trades increased disk reads for
      lower memory usage.

  Returns:
    A Tensor of shape `[num_rows_to_load + num_row_oov_buckets,
    new_col_vocab_size + num_col_oov_buckets]`, with values loaded from the
    specified tensor in the checkpoint, and any missing or OOV values
    initialized with the given `initializer`.

  Raises:
    ValueError: If `num_row_oov_buckets` or `num_col_oov_buckets` < 0.
    ValueError: If either `old_row_vocab_file` or `new_row_vocab_file` is
      provided, while the other is not. Same for `old_col_vocab_file` and
      `new_col_vocab_file`.
    ValueError: If neither row vocabs or col vocabs are provided.
  """
    if num_row_oov_buckets < 0:
        raise ValueError('num_row_oov_buckets must be >= 0, but received %d' % num_row_oov_buckets)
    if num_col_oov_buckets < 0:
        raise ValueError('num_col_oov_buckets must be >= 0, but received %d' % num_col_oov_buckets)
    if bool(old_row_vocab_file) != bool(new_row_vocab_file):
        raise ValueError("old_row_vocab_file and new_row_vocab_file must both be specified or left unspecified. old_row_vocab_file='{}', new_row_vocab_file='{}'".format(old_row_vocab_file, new_row_vocab_file))
    if bool(old_col_vocab_file) != bool(new_col_vocab_file):
        raise ValueError("old_col_vocab_file and new_col_vocab_file must both be specified or left unspecified. old_col_vocab_file='{}', new_col_vocab_file='{}'".format(old_col_vocab_file, new_col_vocab_file))
    remap_rows = new_row_vocab_file and old_row_vocab_file
    remap_cols = new_col_vocab_file and old_col_vocab_file
    if not (remap_rows or remap_cols):
        raise ValueError('Must provide either row or column vocab files. If no remapping is necessary, consider using `tf.contrib.framework.init_from_checkpoint` instead.')
    num_rows_present = num_rows_to_load
    if remap_rows:
        row_remapping, num_rows_present = gen_checkpoint_ops.generate_vocab_remapping(new_vocab_file=new_row_vocab_file, old_vocab_file=old_row_vocab_file, new_vocab_offset=new_row_vocab_offset, num_new_vocab=num_rows_to_load, old_vocab_size=old_row_vocab_size)
    else:
        row_remapping = math_ops.range(new_row_vocab_offset, new_row_vocab_offset + num_rows_to_load, dtype=dtypes.int64)
    col_remapping = []
    num_cols_present = new_col_vocab_size
    if remap_cols:
        col_remapping, num_cols_present = gen_checkpoint_ops.generate_vocab_remapping(new_vocab_file=new_col_vocab_file, old_vocab_file=old_col_vocab_file, new_vocab_offset=0, num_new_vocab=new_col_vocab_size)
    init_vals = initializer([num_rows_to_load * new_col_vocab_size - num_rows_present * num_cols_present, 1])
    return_tensor = gen_checkpoint_ops.load_and_remap_matrix(ckpt_path=ckpt_path, old_tensor_name=old_tensor_name, row_remapping=row_remapping, col_remapping=col_remapping, initializing_values=init_vals, num_rows=num_rows_to_load, num_cols=new_col_vocab_size, max_rows_in_memory=max_rows_in_memory)
    if num_row_oov_buckets > 0:
        init_row_oov_val = initializer([num_row_oov_buckets, new_col_vocab_size])
        init_row_oov_val = ops.convert_to_tensor(init_row_oov_val)
        return_tensor = array_ops.concat([return_tensor, init_row_oov_val], 0)
    if num_col_oov_buckets > 0:
        init_col_oov_val = initializer([num_rows_to_load + num_row_oov_buckets, num_col_oov_buckets])
        init_col_oov_val = ops.convert_to_tensor(init_col_oov_val)
        return_tensor = array_ops.concat([return_tensor, init_col_oov_val], 1)
    return return_tensor