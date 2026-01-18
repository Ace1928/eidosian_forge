from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, cast
from ..config import registry
from ..model import Model
from ..types import ArrayXd, ListXd
@registry.layers('with_flatten.v1')
def with_flatten(layer: Model[InnerInT[ItemT], InnerOutT]) -> Model[InT[ItemT], OutT]:
    return Model(f'with_flatten({layer.name})', forward, layers=[layer], init=init)