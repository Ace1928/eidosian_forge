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
def _cut_circuit_mc_expand(tape: QuantumTape, classical_processing_fn: Optional[callable]=None, max_depth: int=1, shots: Optional[int]=None, device_wires: Optional[Wires]=None, auto_cutter: Union[bool, Callable]=False, **kwargs) -> (Sequence[QuantumTape], Callable):
    """Main entry point for expanding operations in sample-based tapes until
    reaching a depth that includes :class:`~.WireCut` operations."""

    def processing_fn(res):
        return res[0]
    return ([_qcut_expand_fn(tape, max_depth, auto_cutter)], processing_fn)