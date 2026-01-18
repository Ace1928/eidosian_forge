from typing import Callable, List, Optional, Tuple, TypeVar, cast
from ..config import registry
from ..model import Model
from ..types import Array2d, Array3d
@registry.layers('with_reshape.v1')
def with_reshape(layer: Model[OutT, OutT]) -> Model[InT, InT]:
    """Reshape data on the way into and out from a layer."""
    return Model(f'with_reshape({layer.name})', forward, init=init, layers=[layer], dims={'nO': None, 'nI': None})