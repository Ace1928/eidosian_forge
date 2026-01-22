from ... import symbol, ndarray
from ...base import string_types, numeric_types, _as_list
from ..block import Block, HybridBlock
from ..utils import _indent
from .. import tensor_types
from ..nn import LeakyReLU
def _cells_begin_state(cells, **kwargs):
    return sum([c.begin_state(**kwargs) for c in cells], [])