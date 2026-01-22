import string
from typing import List, Sequence
from networkx import MultiDiGraph
import pennylane as qml
from pennylane import numpy as pnp
from .utils import MeasureNode, PrepareNode
def _reshape_results(results: Sequence, shots: int) -> List[List]:
    """
    Helper function to reshape ``results`` into a two-dimensional nested list whose number of rows
    is determined by the number of shots and whose number of columns is determined by the number of
    cuts.
    """
    results = [qml.math.stack(tape_res) if isinstance(tape_res, tuple) else tape_res for tape_res in results]
    results = [qml.math.flatten(r) for r in results]
    results = [results[i:i + shots] for i in range(0, len(results), shots)]
    results = list(map(list, zip(*results)))
    return results