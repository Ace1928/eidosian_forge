from __future__ import annotations
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.passmanager import StagedPassManager
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import (
Level 1 pass manager: light optimization by simple adjacent gate collapsing.

    This pass manager applies the user-given initial layout. If none is given,
    and a trivial layout (i-th virtual -> i-th physical) makes the circuit fit
    the coupling map, that is used.
    Otherwise, the circuit is mapped to the most densely connected coupling subgraph,
    and swaps are inserted to map. Any unused physical qubit is allocated as ancilla space.
    The pass manager then unrolls the circuit to the desired basis, and transforms the
    circuit to match the coupling map. Finally, optimizations in the form of adjacent
    gate collapse and redundant reset removal are performed.

    Args:
        pass_manager_config: configuration of the pass manager.

    Returns:
        a level 1 pass manager.

    Raises:
        TranspilerError: if the passmanager config is invalid.
    