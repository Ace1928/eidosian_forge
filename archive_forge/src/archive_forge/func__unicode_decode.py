import typing
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import map_fn as map_fn_lib
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import compat as util_compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _unicode_decode(input, input_encoding, errors, replacement_char, replace_control_characters, with_offsets):
    """Decodes each string into a sequence of codepoints."""
    input = ragged_tensor.convert_to_tensor_or_ragged_tensor(input, name='input')
    input_ndims = input.shape.ndims
    if input_ndims is None:
        raise ValueError('Rank of `input` must be statically known.')
    if input_ndims > 1:
        if not ragged_tensor.is_ragged(input):
            input = ragged_tensor.RaggedTensor.from_tensor(input, ragged_rank=input_ndims - 1)
        elif input.ragged_rank < input_ndims - 1:
            input = input.with_flat_values(ragged_tensor.RaggedTensor.from_tensor(input.flat_values, ragged_rank=input_ndims - input.ragged_rank - 1))
    if ragged_tensor.is_ragged(input):
        flat_input = array_ops.reshape(input.flat_values, [-1])
    else:
        flat_input = array_ops.reshape(input, [-1])
    if with_offsets:
        decode_op = gen_string_ops.unicode_decode_with_offsets
    else:
        decode_op = gen_string_ops.unicode_decode
    flat_result = decode_op(input=flat_input, input_encoding=input_encoding, errors=errors, replacement_char=replacement_char, replace_control_characters=replace_control_characters)
    if input_ndims == 0:
        codepoints = flat_result.char_values
        if with_offsets:
            offsets = flat_result.char_to_byte_starts
    else:
        codepoints = ragged_tensor.RaggedTensor.from_row_splits(flat_result.char_values, flat_result.row_splits, validate=False)
        if input_ndims > 1:
            codepoints = input.with_flat_values(codepoints)
        if with_offsets:
            offsets = ragged_tensor.RaggedTensor.from_row_splits(flat_result.char_to_byte_starts, flat_result.row_splits, validate=False)
            if input_ndims > 1:
                offsets = input.with_flat_values(offsets)
    if with_offsets:
        return (codepoints, offsets)
    else:
        return codepoints