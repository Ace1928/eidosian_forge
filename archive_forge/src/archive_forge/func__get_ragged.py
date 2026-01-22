from typing import Callable, List, Optional, Tuple, TypeVar, Union, cast
from ..backends import NumpyOps
from ..config import registry
from ..model import Model
from ..types import Array2d, Ints1d, List2d, ListXd, Padded, Ragged
def _get_ragged(model: Model[SeqT, SeqT], seq: SeqT) -> Ragged:
    if isinstance(seq, Ragged):
        return seq
    elif isinstance(seq, Padded):
        lists = model.ops.padded2list(seq)
        lengths = model.ops.asarray1i([len(x) for x in lists])
        k = model.ops.flatten(lists)
        return Ragged(model.ops.flatten(lists), lengths)
    elif _is_ragged_data(seq):
        return Ragged(*seq)
    else:
        list2d_seq = cast(List2d, seq)
        lengths = model.ops.asarray1i([len(x) for x in list2d_seq])
        return Ragged(model.ops.flatten(list2d_seq), lengths)