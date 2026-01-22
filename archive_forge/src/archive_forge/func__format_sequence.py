from ... import symbol, ndarray
from ...base import string_types, numeric_types, _as_list
from ..block import Block, HybridBlock
from ..utils import _indent
from .. import tensor_types
from ..nn import LeakyReLU
def _format_sequence(length, inputs, layout, merge, in_layout=None):
    assert inputs is not None, 'unroll(inputs=None) has been deprecated. Please create input variables outside unroll.'
    axis = layout.find('T')
    batch_axis = layout.find('N')
    batch_size = 0
    in_axis = in_layout.find('T') if in_layout is not None else axis
    if isinstance(inputs, symbol.Symbol):
        F = symbol
        if merge is False:
            assert len(inputs.list_outputs()) == 1, "unroll doesn't allow grouped symbol as input. Please convert to list with list(inputs) first or let unroll handle splitting."
            inputs = list(symbol.split(inputs, axis=in_axis, num_outputs=length, squeeze_axis=1))
    elif isinstance(inputs, ndarray.NDArray):
        F = ndarray
        batch_size = inputs.shape[batch_axis]
        if merge is False:
            assert length is None or length == inputs.shape[in_axis]
            inputs = _as_list(ndarray.split(inputs, axis=in_axis, num_outputs=inputs.shape[in_axis], squeeze_axis=1))
    else:
        assert length is None or len(inputs) == length
        if isinstance(inputs[0], symbol.Symbol):
            F = symbol
        else:
            F = ndarray
            batch_size = inputs[0].shape[0]
        if merge is True:
            inputs = F.stack(*inputs, axis=axis)
            in_axis = axis
    if isinstance(inputs, tensor_types) and axis != in_axis:
        inputs = F.swapaxes(inputs, dim1=axis, dim2=in_axis)
    return (inputs, axis, F, batch_size)