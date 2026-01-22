from typing import Callable, List, Optional, Tuple, TypeVar, Union, cast
from ..backends import NumpyOps
from ..config import registry
from ..model import Model
from ..types import Array2d, Ints1d, List2d, ListXd, Padded, Ragged
def _tuple_forward(layer: Model[Ragged, Ragged], X: RaggedData, is_train: bool) -> Tuple[RaggedData, Callable]:
    Yr, get_dXr = layer(Ragged(*X), is_train)

    def backprop(dY: RaggedData) -> RaggedData:
        dXr = get_dXr(Ragged(*dY))
        return (dXr.data, dXr.lengths)
    return ((Yr.data, Yr.lengths), backprop)