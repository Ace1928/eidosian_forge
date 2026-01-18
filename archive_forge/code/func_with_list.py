from typing import Callable, List, Optional, Tuple, TypeVar, Union, cast
from ..config import registry
from ..model import Model
from ..types import Array2d, Floats2d, Ints2d, List2d, Padded, Ragged
@registry.layers('with_list.v1')
def with_list(layer: Model[List2d, List2d]) -> Model[SeqT, SeqT]:
    return Model(f'with_list({layer.name})', forward, init=init, layers=[layer], dims={name: layer.maybe_get_dim(name) for name in layer.dim_names})