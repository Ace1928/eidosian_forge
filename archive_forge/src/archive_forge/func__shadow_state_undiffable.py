import warnings
from functools import reduce, partial
from itertools import product
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.measurements import ClassicalShadowMP
from pennylane.tape import QuantumScript, QuantumTape
from pennylane import transform
def _shadow_state_undiffable(tape, wires):
    """Non-differentiable version of the shadow state transform"""
    wires_list = wires if isinstance(wires[0], list) else [wires]

    def post_processing(results):
        bits, recipes = results[0]
        shadow = qml.shadows.ClassicalShadow(bits, recipes)
        states = [qml.math.mean(shadow.global_snapshots(wires=w), 0) for w in wires_list]
        return states if isinstance(wires[0], list) else states[0]
    return ([tape], post_processing)