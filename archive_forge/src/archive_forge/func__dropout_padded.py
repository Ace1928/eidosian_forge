from typing import Callable, List, Sequence, Tuple, TypeVar, Union, cast
from ..config import registry
from ..model import Model
from ..types import ArrayXd, Padded, Ragged
def _dropout_padded(model: Model[InT, InT], Xp: Padded, is_train: bool) -> Tuple[Padded, Callable]:
    X = Xp.data
    mask = model.ops.get_dropout_mask(X.shape, model.attrs['dropout_rate'])
    Y = X * mask

    def backprop(dYp: Padded) -> Padded:
        return Padded(dYp.data * mask, dYp.size_at_t, dYp.lengths, dYp.indices)
    return (Padded(Y, Xp.size_at_t, Xp.lengths, Xp.indices), backprop)