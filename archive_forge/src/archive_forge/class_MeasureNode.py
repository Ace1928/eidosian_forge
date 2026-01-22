import uuid
from typing import Any, Callable, Sequence, Tuple
import warnings
import numpy as np
from networkx import MultiDiGraph, has_path, weakly_connected_components
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.ops.meta import WireCut
from pennylane.queuing import WrappedObj
from pennylane.operation import Operation
from .kahypar import kahypar_cut
from .cutstrategy import CutStrategy
class MeasureNode(Operation):
    """Placeholder node for measurement operations"""
    num_wires = 1
    grad_method = None

    def __init__(self, *params, wires=None, id=None):
        id = id or str(uuid.uuid4())
        super().__init__(*params, wires=wires, id=id)