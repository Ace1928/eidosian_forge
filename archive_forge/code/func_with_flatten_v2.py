from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, cast
from ..config import registry
from ..model import Model
@registry.layers('with_flatten.v2')
def with_flatten_v2(layer: Model[FlatT[InItemT], FlatT[OutItemT]]) -> Model[NestedT[InItemT], NestedT[OutItemT]]:
    return Model(f'with_flatten({layer.name})', forward, layers=[layer], init=init)