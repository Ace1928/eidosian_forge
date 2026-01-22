from itertools import islice, product
from typing import List
import numpy as np
import pennylane as qml
from pennylane import BasisState, QubitDevice, StatePrep
from pennylane.devices import DefaultQubitLegacy
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operation
from pennylane.wires import Wires
from ._serialize import QuantumScriptSerializer
from ._version import __version__
def _chunk_iterable(iteration, num_chunks):
    """Lazy-evaluated chunking of given iterable from https://stackoverflow.com/a/22045226"""
    iteration = iter(iteration)
    return iter(lambda: tuple(islice(iteration, num_chunks)), ())