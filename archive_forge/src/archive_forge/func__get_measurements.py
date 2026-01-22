import copy
from itertools import product
from typing import Callable, List, Sequence, Tuple, Union
from networkx import MultiDiGraph
import pennylane as qml
from pennylane import expval
from pennylane.measurements import ExpectationMP, MeasurementProcess, SampleMP
from pennylane.operation import Operator, Tensor
from pennylane.ops.meta import WireCut
from pennylane.pauli import string_to_pauli_word
from pennylane.queuing import WrappedObj
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.wires import Wires
from .utils import MeasureNode, PrepareNode
def _get_measurements(group: Sequence[Operator], measurements: Sequence[MeasurementProcess]) -> List[MeasurementProcess]:
    """Pairs each observable in ``group`` with the circuit ``measurements``.

    Only a single measurement of an expectation value is currently supported
    in ``measurements``.

    Args:
        group (Sequence[Operator]): a collection of observables
        measurements (Sequence[MeasurementProcess]): measurements from the circuit

    Returns:
        List[MeasurementProcess]: the expectation values of ``g @ obs``, where ``g`` is iterated
        over ``group`` and ``obs`` is the observable composing the single measurement
        in ``measurements``
    """
    if len(group) == 0:
        return measurements
    n_measurements = len(measurements)
    if n_measurements > 1:
        raise ValueError('The circuit cutting workflow only supports circuits with a single output measurement')
    if n_measurements == 0:
        return [expval(g) for g in group]
    measurement = measurements[0]
    if not isinstance(measurement, ExpectationMP):
        raise ValueError('The circuit cutting workflow only supports circuits with expectation value measurements')
    obs = measurement.obs
    return [expval(copy.copy(obs) @ g) for g in group]