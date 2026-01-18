from typing import List, Sequence, Callable
import pennylane as qml
from pennylane.measurements import ExpectationMP, MeasurementProcess
from pennylane.ops import SProd, Sum
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms import transform
Group observables of ``measurements`` into groups with non overlapping wires.

    Args:
        measurements (List[MeasurementProcess]): list of measurement processes

    Returns:
        List[List[MeasurementProcess]]: list of groups of observables with non overlapping wires
    