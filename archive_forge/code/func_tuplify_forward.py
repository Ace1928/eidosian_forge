from typing import Any, Optional, Tuple, TypeVar
from ..config import registry
from ..model import Model
def tuplify_forward(model, X, is_train):
    Ys = []
    backprops = []
    for layer in model.layers:
        Y, backprop = layer(X, is_train)
        Ys.append(Y)
        backprops.append(backprop)

    def backprop_tuplify(dYs):
        dXs = [bp(dY) for bp, dY in zip(backprops, dYs)]
        dX = dXs[0]
        for dx in dXs[1:]:
            dX += dx
        return dX
    return (tuple(Ys), backprop_tuplify)