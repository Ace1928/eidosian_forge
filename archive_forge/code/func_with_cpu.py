from typing import Any, Callable, Tuple
import numpy
from thinc.backends import Ops
from ..config import registry
from ..model import Model
@registry.layers('with_cpu.v1')
def with_cpu(layer: Model, ops: Ops) -> Model:
    layer.to_cpu()
    return Model(f'with_cpu({layer.name})', forward, layers=[layer], ops=ops, init=init, dims={name: layer.maybe_get_dim(name) for name in layer.dim_names})