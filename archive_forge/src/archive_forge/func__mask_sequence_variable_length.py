from ... import symbol, ndarray
from ...base import string_types, numeric_types, _as_list
from ..block import Block, HybridBlock
from ..utils import _indent
from .. import tensor_types
from ..nn import LeakyReLU
def _mask_sequence_variable_length(F, data, length, valid_length, time_axis, merge):
    assert valid_length is not None
    if not isinstance(data, tensor_types):
        data = F.stack(*data, axis=time_axis)
    outputs = F.SequenceMask(data, sequence_length=valid_length, use_sequence_length=True, axis=time_axis)
    if not merge:
        outputs = _as_list(F.split(outputs, num_outputs=length, axis=time_axis, squeeze_axis=True))
    return outputs