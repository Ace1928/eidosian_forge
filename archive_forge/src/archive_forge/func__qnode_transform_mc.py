import inspect
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
from networkx import MultiDiGraph
import pennylane as qml
from pennylane.measurements import SampleMP
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms import transform
from pennylane.wires import Wires
from .cutstrategy import CutStrategy
from .kahypar import kahypar_cut
from .processing import qcut_processing_fn_mc, qcut_processing_fn_sample
from .tapes import _qcut_expand_fn, graph_to_tape, tape_to_graph
from .utils import (
@cut_circuit_mc.custom_qnode_transform
def _qnode_transform_mc(self, qnode, targs, tkwargs):
    """Here, we overwrite the QNode execution wrapper in order
    to access the device wires."""
    if tkwargs.get('shots', False):
        raise ValueError("Cannot provide a 'shots' value directly to the cut_circuit_mc decorator when transforming a QNode. Please provide the number of shots in the device or when calling the QNode.")
    if 'shots' in inspect.signature(qnode.func).parameters:
        raise ValueError("Detected 'shots' as an argument of the quantum function to transform. The 'shots' argument name is reserved for overriding the number of shots taken by the device.")
    tkwargs.setdefault('device_wires', qnode.device.wires)
    execute_kwargs = getattr(qnode, 'execute_kwargs', {}).copy()
    execute_kwargs['cache'] = False
    new_qnode = self.default_qnode_transform(qnode, targs, tkwargs)
    new_qnode.__class__ = CustomQNode
    new_qnode.execute_kwargs = execute_kwargs
    return new_qnode