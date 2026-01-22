from typing import Sequence, Tuple, TypeVar, Union
from ..model import Model
from ..types import ArrayXd, FloatsXd, IntsXd
def array_getitem(index: Index) -> Model[ArrayTXd, ArrayTXd]:
    """Index into input arrays, and return the subarrays.

    index:
        A valid numpy-style index. Multi-dimensional indexing can be performed
        by passing in a tuple, and slicing can be performed using the slice object.
        For instance, X[:, :-1] would be (slice(None, None), slice(None, -1)).
    """
    return Model('array-getitem', forward, attrs={'index': index})