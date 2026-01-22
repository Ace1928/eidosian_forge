from typing import Callable, List, Optional, Tuple, TypeVar, Union, cast
from ..config import registry
from ..model import Model
from ..types import Array2d, Floats3d, Ints1d, List2d, Padded, Ragged
from ..util import is_xp_array
def _get_padded(model: Model, seq: SeqT) -> Padded:
    if isinstance(seq, Padded):
        return seq
    elif isinstance(seq, Ragged):
        return model.ops.list2padded(model.ops.unflatten(seq.data, seq.lengths))
    elif _is_padded_data(seq):
        return Padded(*seq)
    elif is_xp_array(seq):
        floats3d_seq = cast(Floats3d, seq)
        size_at_t = model.ops.asarray1i([floats3d_seq.shape[1]] * floats3d_seq.shape[0])
        lengths = model.ops.asarray1i([floats3d_seq.shape[0]] * floats3d_seq.shape[1])
        indices = model.ops.xp.arange(floats3d_seq.shape[1])
        return Padded(floats3d_seq, size_at_t, lengths, indices)
    else:
        assert isinstance(seq, list), seq
        return model.ops.list2padded(seq)