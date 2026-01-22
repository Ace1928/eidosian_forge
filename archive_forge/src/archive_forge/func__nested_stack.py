from typing import Callable, Sequence
import pennylane as qml
from .core import transform
def _nested_stack(res):
    """
    Given a list of identical nested tuple structures, stack the arrays at the leaves
    """
    if not isinstance(res[0], (tuple, qml.numpy.builtins.SequenceBox)):
        return qml.math.stack(res)
    stacked_results = []
    for i in range(len(res[0])):
        stacked_results.append(_nested_stack([r[i] for r in res]))
    return tuple(stacked_results)