from ... import symbol, ndarray
from ...base import string_types, numeric_types, _as_list
from ..block import Block, HybridBlock
from ..utils import _indent
from .. import tensor_types
from ..nn import LeakyReLU
def _cells_state_info(cells, batch_size):
    return sum([c.state_info(batch_size) for c in cells], [])