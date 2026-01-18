from typing import Any, Optional, Tuple, TypeVar
from ..config import registry
from ..model import Model
@registry.layers('tuplify.v1')
def tuplify(layer1: Model[InT, Any], layer2: Model[InT, Any], *layers) -> Model[InT, Tuple]:
    """Send a separate copy of the input to each child layer, and join the
    outputs of the children into a tuple on the way out.

    Typically used to provide both modified data and the original input to a
    downstream layer.
    """
    layers = (layer1, layer2) + layers
    names = [layer.name for layer in layers]
    return Model('tuple(' + ', '.join(names) + ')', tuplify_forward, init=init, layers=layers, dims={'nI': None})