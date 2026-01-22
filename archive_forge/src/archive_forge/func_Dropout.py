from typing import Callable, List, Sequence, Tuple, TypeVar, Union, cast
from ..config import registry
from ..model import Model
from ..types import ArrayXd, Padded, Ragged
@registry.layers('Dropout.v1')
def Dropout(rate: float=0.0) -> Model[InT, InT]:
    """Help prevent overfitting by adding a random distortion to the input data
    during training.  Specifically, cells of the input are zeroed with
    probability determined by the `rate` argument.
    """
    return Model('dropout', forward, attrs={'dropout_rate': rate, 'is_enabled': True})